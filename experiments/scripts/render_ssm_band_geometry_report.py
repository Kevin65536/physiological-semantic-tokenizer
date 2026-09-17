#!/usr/bin/env python3
"""Read-only structural SSM analysis; fixed denominators and bitmap PDF figures."""
from __future__ import annotations
import argparse
import base64
import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
import markdown
import yaml

CANDIDATES=['B_ref','local_scalar','local_multi','merged','split','geometry','permuted']
MODES=['full','center_EEG','center_fNIRS','fNIRS_only','EEG_only','all_missing']
STATES=['r','s','f','v','p','q']


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def hierarchical(rows,field):
    sessions=defaultdict(list)
    for row in rows:
        if row.get(field) is not None:sessions[(row['subject'],row['session'])].append(row[field])
    subjects=defaultdict(list)
    for (subject,_),values in sessions.items():subjects[subject].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in subjects.values()])) if subjects else None


def risk(rows,expected=72):
    center={r['sample_id']:r for r in rows if r['mode']=='center_fNIRS' and r['status']=='completed' and 'mse' in r}
    whole={r['sample_id']:r for r in rows if r['mode']=='EEG_only' and r['status']=='completed' and 'mse' in r}
    common=set(center)&set(whole)
    matched=[dict(center[k],risk=(center[k]['mse']+whole[k]['mse'])/2) for k in common]
    return dict(valid_center=len(center),valid_whole=len(whole),paired_identities=len(common),expected=expected,
                RF=hierarchical(matched,'risk') if len(common)==expected else None,
                common_success_subset_RF=hierarchical(matched,'risk'))


def paired_interval(values):
    if len(values)!=4:return dict(mean=None,upper95=None,panels=len(values))
    values=np.asarray(values)
    return dict(mean=float(values.mean()),upper95=float(values.mean()+student_t.ppf(.95,3)*values.std(ddof=1)/2),panels=4)


def analyze(run, *, partial=False, template_correction=None):
    csv.field_size_limit(20_000_000)
    tasks=[json.loads(r['payload']) for r in csv.DictReader((run/'task_table.csv').open())]
    rows=[];failures=[];cells=[];starts=[];corrections=[];corrected={}
    if template_correction is not None:
        template_correction=Path(template_correction).resolve()
        manifest=read(template_correction/'manifest.json')
        if manifest['execution']!='completed' or Path(manifest['source_run']).resolve()!=run.resolve():
            raise ValueError('completed correction must refer to this exact original run')
        fixed_tasks=[json.loads(r['payload']) for r in csv.DictReader((template_correction/'task_table.csv').open())]
        expected={t['id'] for t in tasks if t['family']=='M' and t['kind']=='structure_fit'}
        if {t['id'] for t in fixed_tasks}!=expected or len(fixed_tasks)!=504:
            raise ValueError('correction identity denominator differs from original 504 fits')
        for task in fixed_tasks:
            result=read(template_correction/'cells'/task['id']/'result.json')
            if len(result.get('rows',[]))!=1 or result['rows'][0]['mode']!='EEG_only_template':
                raise ValueError('correction row missing; no fallback to the original control')
            corrected[task['id']]=result['rows'][0]
            cells.append(dict(task_id=task['id'],kind='template_covariance_correction',family='R',status=result['status'],
                elapsed=result.get('elapsed_seconds'),worker_peak_rss_bytes=result.get('peak_rss_bytes'),solves=result.get('actual_solves',0)))
    for task in tasks:
        path=run/'cells'/task['id']/'result.json'
        result=(dict(status='pending',rows=[]) if partial and not path.exists() else read(path))
        cells.append(dict(task_id=task['id'],kind=task['kind'],status=result['status'],family=task['family'],
                          elapsed=result.get('elapsed_seconds'),worker_peak_rss_bytes=result.get('peak_rss_bytes'),
                          solves=result.get('actual_solves',0),reason=result.get('reason',result.get('error'))))
        for value in result.get('rows',[]):
            if 'mode' not in value:continue
            corrected_source=False
            if task['id'] in corrected and value['mode']=='EEG_only_template':
                replacement=corrected[task['id']]
                corrections.append(dict(task_id=task['id'],candidate=value['candidate_name'],original_status=value['status'],
                    corrected_status=replacement['status'],original_fnirs_mse=value.get('fnirs_mse'),corrected_fnirs_mse=replacement.get('fnirs_mse'),
                    original_run=str(run),corrected_run=str(template_correction)))
                value=replacement;corrected_source=True
            row={k:v for k,v in value.items() if k in ('sample_id','subject','session','trial','outer','candidate_name','mode','control','status','condition','panel','fnirs_mse','fnirs_bias','state_nrmse','r_correlation','clean_hb_nrmse','clean_eeg_common_nrmse','common_eeg_audit_nrmse','retained_rank','objective','selected_start')}
            row.update(family=('SL' if task.get('synthetic') and task['family']=='L' else task['family']),task_id=task['id'])
            row['result_source']=str(template_correction if corrected_source else run)
            if task.get('synthetic') and task['family']=='L':
                detail=read(run/'prepared'/f"{task['subject']}.json")
                row.update(condition=detail['condition'],panel=detail['panel'])
            if 'fnirs_mse' in row:
                row.update(mse=float(np.mean(row['fnirs_mse'])),hbo_mse=row['fnirs_mse'][0],hbr_mse=row['fnirs_mse'][1])
            metrics=value.get('center_metrics' if row['mode'].startswith('center_fNIRS') else 'metrics',{})
            for label in ('HbO','HbR'):
                if metrics.get(label):
                    row[label+'_abs_error_quantiles_sd']=metrics[label].get('absolute_residual_sd_quantiles')
            if row['status']=='completed' and row['mode']=='center_EEG':
                with np.load(run/'cells'/task['id']/(row['mode']+'.npz')) as arrays:
                    audit=arrays['r']-arrays['r'][:20].mean()
                    row['hidden_eeg_audit_nrmse']=float(np.sqrt(np.mean((audit[52:68]-arrays['target'][52:68,0])**2))/arrays['normalization_sd'][0])
            elif row['status']=='completed' and row['mode']=='fNIRS_only':
                row['hidden_eeg_audit_nrmse']=row.get('common_eeg_audit_nrmse')
            if row['mode']=='full':
                row['full_clean_hb_residual_nrmse']=[(value.get('clean_map_residual',{}).get(m) or {}).get('nrmse') for m in ('HbO','HbR')]
            rows.append(row)
            if row['status']!='completed':failures.append(dict(row,error=value.get('error'),failure_category=value.get('failure_category')))
            for attempt in value.get('starts',[]):
                starts.append(dict(task_id=task['id'],candidate=row['candidate_name'],mode=row['mode'],family=row['family'],status=attempt['status'],
                    reason=attempt.get('convergence_reason'),evaluations=attempt.get('evaluations'),
                    rejected_domain_steps=sum(r.get('kind')=='domain_trial_step' for r in attempt.get('rejected_steps',[]))))
    syn=[]
    for condition in sorted({r.get('condition') for r in rows if r['family']=='S'}):
        for candidate in CANDIDATES:
            for mode in sorted({r['mode'] for r in rows if r['family']=='S' and r['condition']==condition}):
                group=[r for r in rows if r['family']=='S' and r['condition']==condition and r['candidate_name']==candidate and r['mode']==mode]
                if not group:continue
                good=[r for r in group if r['status']=='completed' and 'state_nrmse' in r]
                record=dict(condition=condition,candidate=candidate,mode=mode,expected=24,valid=len(good),statuses=dict(Counter(r['status'] for r in group)))
                for key in ('state_nrmse','r_correlation','clean_hb_nrmse','clean_eeg_common_nrmse','mse'):
                    vals=[r[key] for r in good if key in r]
                    record[key]=np.mean(vals,axis=0).tolist() if vals else None
                syn.append(record)
    noninferiority=[]
    lookup={(r['condition'],r['panel'],r['trial'],r['candidate_name'],r['mode']):r for r in rows if r['family']=='S'}
    for candidate in CANDIDATES[1:]:
        for label,field,j in [(x,'state_nrmse',i) for i,x in enumerate(STATES)]+[('HbO','clean_hb_nrmse',0),('HbR','clean_hb_nrmse',1)]:
            panel_values=[]
            for panel in range(4):
                differences=[]
                for trial in range(18,24):
                    for mode in MODES:
                        a=lookup.get(('shared',panel,trial,candidate,mode));b=lookup.get(('shared',panel,trial,'B_ref',mode))
                        if a and b and a['status']==b['status']=='completed' and field in a and field in b:differences.append(a[field][j]-b[field][j])
                if len(differences)==36:panel_values.append(float(np.mean(differences)))
            interval=paired_interval(panel_values)
            noninferiority.append(dict(candidate=candidate,coordinate=label,**interval,passed=interval['upper95'] is not None and interval['upper95']<=.02))
    stress_state_comparisons=[]
    for condition in sorted({r['condition'] for r in rows if r['family']=='S' and r['condition']!='shared'}):
        for candidate in CANDIDATES[1:]:
            differences=[]
            for panel in range(4):
                values=[]
                for trial in range(18,24):
                    a=lookup.get((condition,panel,trial,candidate,'full'));b=lookup.get((condition,panel,trial,'B_ref','full'))
                    if a and b and a['status']==b['status']=='completed':values.append(a['state_nrmse'][0]-b['state_nrmse'][0])
                if len(values)==6:differences.append(float(np.mean(values)))
            interval=paired_interval(differences)
            stress_state_comparisons.append(dict(condition=condition,candidate=candidate,coordinate='r',mode='full',**interval))
    measured=[];by_subject=[];completion=[];modes=[]
    for family in ('M','L'):
        for candidate in CANDIDATES:
            controls=['correct','context','template','own','pairing','shift'] if family=='L' else ['correct']
            for control in controls:
                group=[r for r in rows if r['family']==family and r['candidate_name']==candidate and (family!='L' or r['control']==control)]
                if not group:continue
                measured.append(dict(family=family,candidate=candidate,control=control,**risk(group)))
                for subject in ('subject_01','subject_09','subject_18'):
                    by_subject.append(dict(family=family,candidate=candidate,control=control,subject=subject,**risk([r for r in group if r['subject']==subject],24)))
                for mode in (MODES if family=='M' else ['center_fNIRS','EEG_only']):
                    part=[r for r in group if r['mode']==mode];good=[r for r in part if r['status']=='completed' and 'mse' in r]
                    modes.append(dict(family=family,candidate=candidate,control=control,mode=mode,expected=72,valid=len(good),
                        mse=hierarchical(good,'mse'),hbo_nrmse=np.sqrt(hierarchical(good,'hbo_mse')) if good else None,
                        hbr_nrmse=np.sqrt(hierarchical(good,'hbr_mse')) if good else None))
                    if family=='M':completion.append(dict(candidate=candidate,mode=mode,expected=72,attempted=len(part),valid=len(good),statuses=dict(Counter(r['status'] for r in part))))
    paired=[]
    for family in ('S','M','SL','L'):
        candidates=CANDIDATES if family in ('S','M') else CANDIDATES[:5]
        conditions=sorted({r.get('condition') for r in rows if r['family']==family}) if family in ('S','SL') else [None]
        for condition in conditions:
            for candidate in candidates:
                for mode in ('center_fNIRS','EEG_only'):
                    for null in ('pairing','shift','template','own','context'):
                        same=[r for r in rows if r['family']==family and r.get('condition')==condition and r['candidate_name']==candidate and r['status']=='completed' and 'mse' in r]
                        correct={r['sample_id']:r for r in same if r['mode']==mode and r.get('control','correct')=='correct'}
                        controls={r['sample_id']:r for r in same if (r['mode']==mode and r.get('control')==null if family in ('L','SL') else r['mode']==mode+'_'+null)}
                        keys=set(correct)&set(controls)
                        if not keys:continue
                        differences=[dict(correct[k],increment=controls[k]['mse']-correct[k]['mse']) for k in keys]
                        paired.append(dict(family=family,condition=condition,candidate=candidate,mode=mode,null=null,valid=len(keys),expected=24 if family in ('S','SL') else 72,
                            increment=hierarchical(differences,'increment'),positive_means='correct pairing better',
                            panel_increments={str(panel):float(np.mean([r['increment'] for r in differences if r.get('panel')==panel])) for panel in range(4) if any(r.get('panel')==panel for r in differences)}))
    geometry=[]
    for subject in ('subject_01','subject_09','subject_18'):
        for outer in range(4):
            for candidate in ('geometry','permuted'):
                path=run/'prepared'/f'{subject}_o{outer}_{candidate}.json'
                if path.exists():
                    j=read(path)['structure_fit'];g=j['geometry']
                    geometry.append(dict(subject=subject,outer=outer,candidate=candidate,penalty=j['selected_penalty'],channels=g['channels'],weights=j['geometry_weights_used'],
                        coordinate_system=g['coordinate_system'],coordinate_units=g['coordinate_units'],loading=j['loading'],noise_floor_fraction=j['noise_audit']['floor_fraction']))
    comparisons=[]
    for family in ('M','L'):
        for candidate,reference in [('local_scalar','B_ref'),('local_multi','local_scalar'),('merged','local_multi'),('split','merged'),('geometry','split'),('geometry','permuted')]:
            for wanted in (('center_fNIRS','EEG_only'),('EEG_only',)):
                groups=[]
                for name in (candidate,reference):
                    subset=[r for r in rows if r['family']==family and r['candidate_name']==name and r['status']=='completed' and 'mse' in r and r.get('control','correct')=='correct']
                    lookup_group={(r['sample_id'],r['mode']):r for r in subset}
                    ids={r['sample_id'] for r in subset if all((r['sample_id'],mode) in lookup_group for mode in wanted)}
                    groups.append((lookup_group,ids))
                common=groups[0][1]&groups[1][1]
                if not common and family=='L' and candidate in ('geometry','permuted'):continue
                values=[]
                for lookup_group,_ in groups:
                    values.append(hierarchical([dict(lookup_group[(i,wanted[0])],score=np.mean([lookup_group[(i,m)]['mse'] for m in wanted])) for i in common],'score'))
                comparisons.append(dict(family=family,candidate=candidate,reference=reference,endpoint='RF' if len(wanted)==2 else 'whole_fNIRS',
                    common_identities=len(common),expected=72,candidate_risk=values[0],reference_risk=values[1],
                    relative_improvement=(1-values[0]/values[1]) if values[1] else None))
    for row in paired:
        values=list(row['panel_increments'].values())
        if len(values)==4 and row['valid']==24:
            width=student_t.ppf(.975,3)*np.std(values,ddof=1)/2
            row['panel_ci95']=[float(np.mean(values)-width),float(np.mean(values)+width)]
        else:row['panel_ci95']=None
    source=Path(yaml.safe_load((run/'resolved_config.yaml').read_text())['structure']['source_run'])
    if not source.is_absolute():source=Path(read(run/'manifest.json')['project_root'])/source
    replay=[]
    for task in tasks:
        if task.get('family')!='M' or task.get('candidate_name')!='B_ref':continue
        path=run/'cells'/task['id']/'result.json'
        if not path.exists():continue
        previous=f"B__prior_gauge__0p0__{task['subject']}__{task['outer']}__{task['trial']}__O2"
        old={r['mode']:r for r in read(source/'cells'/previous/'result.json')['rows']}
        for row in read(path)['rows']:
            if row['mode'] not in old:continue
            prior=old[row['mode']]
            differences=[]
            if row['status']==prior['status']=='completed':
                differences=[abs(row['metrics'][m]['nmse']-prior['metrics'][m]['nmse']) for m in ('HbO','HbR')]
            replay.append(dict(task_id=task['id'],mode=row['mode'],status_identical=row['status']==prior['status'],
                               maximum_hb_nmse_difference=max(differences) if differences else None))
    return dict(rows=rows,synthetic=syn,noninferiority=noninferiority,stress_state_comparisons=stress_state_comparisons,risks=measured,by_subject=by_subject,completion=completion,modes=modes,
                paired_increments=paired,comparisons=comparisons,geometry=geometry,baseline_replay=replay,corrections=corrections,failures=failures,cells=cells,optimizer_starts=starts)


def table(rows,columns):
    def fmt(v):
        if v is None:return '未定义'
        if isinstance(v,(float,np.floating)):return f'{v:.5g}'
        if isinstance(v,(list,dict)):return html.escape(json.dumps(v,ensure_ascii=False))
        return str(v)
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+''.join('| '+' | '.join(fmt(r.get(c)) for c in columns)+' |\n' for r in rows)


def render(run,out,data):
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    shutil.copyfile(__file__,out/'render_source.py')
    dump(out/'analysis.json',data)
    for key,rows in data.items():pd.DataFrame(rows).to_csv(out/f'{key}.csv',index=False)
    figures=[]
    def save(name,fig):
        fig.tight_layout();fig.savefig(out/'figures'/f'{name}.png',dpi=240,bbox_inches='tight');plt.close(fig);figures.append(name)
    fig,ax=plt.subplots(figsize=(8,3.5));width=.11;x=np.arange(len(CANDIDATES))
    for j,mode in enumerate(MODES):
        vals=[next((r['valid'] for r in data['completion'] if r['candidate']==c and r['mode']==mode),0) for c in CANDIDATES]
        ax.bar(x+(j-2.5)*width,vals,width,label=mode)
    ax.set_xticks(x,CANDIDATES,rotation=20);ax.set_ylabel('Valid / fixed 72 trials');ax.set_ylim(0,78);ax.legend(fontsize=7,ncol=3);save('completion',fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    for j,condition in enumerate(('shared','opposite_bands')):
        records=[r for r in data['synthetic'] if r['condition']==condition and r['mode']=='full']
        vals=[next((r['state_nrmse'][0] for r in records if r['candidate']==c and r['state_nrmse']),np.nan) for c in CANDIDATES]
        axs[j].bar(CANDIDATES,vals);axs[j].set_title(condition);axs[j].tick_params(axis='x',rotation=40);axs[j].set_ylabel('Driver NRMSE, successful subset')
    save('synthetic',fig)
    fig,ax=plt.subplots(figsize=(8,3.5));lin=[r for r in data['risks'] if r['family']=='L']
    for j,control in enumerate(('correct','context','template','pairing','shift')):
        vals=[next((r['RF'] for r in lin if r['candidate']==c and r['control']==control),np.nan) for c in CANDIDATES[:5]]
        ax.bar(np.arange(5)+(j-2)*.15,[np.nan if v is None else v for v in vals],.15,label=control)
    ax.set_xticks(np.arange(5),CANDIDATES[:5]);ax.set_ylabel('RF (center / whole fNIRS equal weight)');ax.legend(fontsize=7,ncol=3);save('linear',fig)
    report='# 分频带与空间约束 SSM 开发测试\n\n'
    report+='本报告读取冻结任务表和逐拟合证据；固定六状态、生理参数、过程噪声和 Gaussian 轨迹 MAP。结果只解释为离线条件补全。\n\n'
    complete_ssm=[r for r in data['risks'] if r['family']=='M' and r['RF'] is not None]
    report+=f"本轮形成完整主端点的 SSM 候选为 {len(complete_ssm)}/7。"
    if not complete_ssm:report+='因此不能按预设 10% 主风险改善标准保留新的 SSM；合成收益与整模态预测子项需要分别解释。'
    report+='\n\n'
    synthetic_full={(r['condition'],r['candidate']):r for r in data['synthetic'] if r['mode']=='full'}
    if all(k in synthetic_full for k in [('shared','B_ref'),('shared','split'),('opposite_bands','merged'),('opposite_bands','split')]):
        values=[synthetic_full[k]['state_nrmse'][0] for k in [('shared','B_ref'),('shared','split'),('opposite_bands','merged'),('opposite_bands','split')]]
        report+=f"已知共享驱动下，完整模式 r NRMSE 从宽频基线 {values[0]:.4f} 降至分频带 {values[1]:.4f}；相反频带贡献下，合并频带为 {values[2]:.4f}，分频带为 {values[3]:.4f}。这一合成收益不自动转成实测改进。\n\n"
    linear_risk={(r['candidate'],r['control']):r['RF'] for r in data['risks'] if r['family']=='L'}
    if all(linear_risk.get((c,'correct')) is not None for c in ('B_ref','merged','split')):
        report+=f"完整线性对照的 RF：宽频 {linear_risk['B_ref','correct']:.5f}，合并频带 {linear_risk['merged','correct']:.5f}，分频带 {linear_risk['split','correct']:.5f}。\n\n"
    report+='## 设计与完成情况\n\n'
    report+='3 被试 × 3 session × 8 原训练 trial；4 外折、3 内折。局部最多 6 个 EEG 通道，alpha 8–13、beta 13–30、low-gamma 30–45 Hz。目标 Hb 对及评分尺度在每折各候选间相同。载荷采用训练宽频 PCA 代理回归，几何强度在内折载荷重建 MSE 上选取；本轮没有缺失目标端到端训练。内折重新拟合目标锚点、邻域、缩放与 PCA。\n\n'
    counts=Counter((r['family'],r['status']) for r in data['cells'])
    report+=table([dict(family=k[0],status=k[1],cells=v) for k,v in counts.items()],['family','status','cells'])
    report+='\nS 为合成、G 为入口检查、M 为实测 SSM、L 为线性对照（包含合成及实测）。cell 完成不等于内部每次 MAP 有效；失败始终保留。\n\n'
    if data['corrections']:
        report+=f"另在独立版本补跑 {len(data['corrections'])} 次整段 fNIRS 缺失的 EEG 模板控制（R）：平均 N 个训练 trial 的 EEG 特征，其噪声因子应除以 √N。原始 run 错用单 trial 噪声，此处模板比较采用修正版；原始记录完整保留，corrections.csv 逐项连接新旧结果。主 RF、载荷/正则选择及其余拟合未变。\n\n"
    replay=data['baseline_replay'];deltas=[r['maximum_hb_nmse_difference'] for r in replay if r['maximum_hb_nmse_difference'] is not None]
    report+=f"与上一组新测量基线共有的 {len(replay)}/720 次拟合中，状态一致 {sum(r['status_identical'] for r in replay)} 次；有效行 Hb NMSE 最大差值为 {max(deltas) if deltas else None}。\n\n"
    report+='## 主端点与每名被试\n\n'
    report+='RF 为中心和整段 fNIRS 缺失风险各占一半，内部对 HbO/HbR 训练 SD 归一化 MSE 等权，再按 trial→session→subject 等权。完整端点要求相同 72 身份的两个模式均有效。\n\n'
    report+=table(data['risks'],['family','candidate','control','valid_center','valid_whole','paired_identities','RF','common_success_subset_RF'])
    report+='\n'+table([r for r in data['by_subject'] if r['control']=='correct'],['family','candidate','subject','paired_identities','RF'])
    report+='\n预定关键对照只在候选与参照共同成功的相同身份上比较；少于 72 的比较不升级为完整主端点：\n\n'
    report+=table(data['comparisons'],['family','candidate','reference','endpoint','common_identities','candidate_risk','reference_risk','relative_improvement'])
    report+='\n实测 SSM 各必需模式有效数：\n\n'+table(data['completion'],['candidate','mode','valid','expected'])
    report+='\n## 分模式预测与配对贡献\n\n'
    report+=table([r for r in data['modes'] if r['mode'] in ('center_fNIRS','EEG_only') and r['control']=='correct'],['family','candidate','mode','valid','mse','hbo_nrmse','hbr_nrmse'])
    report+='\n错配和移位只干预 EEG 特征。正增量表示正确配对的风险更低；未完成分母只能作为成功子集诊断。\n\n'
    report+=table([r for r in data['paired_increments'] if r['family'] in ('M','L')],['family','candidate','mode','null','valid','increment'])
    report+='\n## 合成恢复与错误耦合检查\n\n每情形 4 独立 panel，每 panel 18 训练与 6 评价。下表是完整模式的成功子集，所有分母固定 24；六状态及 clean EEG/Hb 全表保存在 synthetic.csv。\n\n'
    small=[]
    for r in data['synthetic']:
        if r['mode']=='full':small.append(dict(condition=r['condition'],candidate=r['candidate'],valid=r['valid'],r_nrmse=r['state_nrmse'][0] if r['state_nrmse'] else None,r_corr=r['r_correlation']))
    report+=table(small,['condition','candidate','valid','r_nrmse','r_corr'])
    report+='\n四 panel 非劣检验：在共享 Gaussian 下，每 panel 对六个评价 trial 和六必需模式平均，再与 B_ref 配对。单侧 95% t 上界≤+0.02 才通过；不以时间点充当独立样本。\n\n'
    report+=table(data['noninferiority'],['candidate','coordinate','panels','mean','upper95','passed'])
    report+='\n失配压力下完整模式 r 恢复相对 B_ref 的四 panel 差值（正值为退化，和上表六模式平均检验分开）：\n\n'
    report+=table(data['stress_state_comparisons'],['condition','candidate','panels','mean','upper95'])
    report+='\n共同任务时序但 trial 扰动独立的整段 fNIRS 缺失 null 结果：\n\n'
    report+=table([r for r in data['paired_increments'] if r['condition'] in ('task_only','task_only_student') and r['mode']=='EEG_only' and r['null']=='pairing'],['family','condition','candidate','valid','increment','panel_ci95'])
    report+='\n四 panel 区间只作开发诊断，未作多重比较校正。错配 donor 来自训练分区，而正确输入来自评价 trial；两者对已拟合线性模型的分布地位不同。已知无试次共享的情形出现小幅正增量时，应将其保留为 null 局限，不能据此声称发现真实耦合。后续可在独立评价 trial 内交换 donor 复核。\n\n'
    report+='无试次共享情形的 r 真值指 fNIRS 的生成驱动；EEG 由另一个具有相同任务均值、独立试次扰动的驱动生成，并不存在跨模态的共有 r 真值。\n\n'
    report+='\n## 几何选择、限制与保留判断\n\n'
    report+=table([dict(candidate=c,penalty=p,folds=sum(r['candidate']==c and r['penalty']==p for r in data['geometry'])) for c in ('geometry','permuted') for p in (0.,.1,1.)],['candidate','penalty','folds'])
    report+='\n几何来自 dataset montage 的共同坐标，仅支持邻近与软约束，不支持个体皮层共配准。若选择零额外惩罚，真实/置换几何退回同一无几何模型，不能把相同结果解释为空间收益。带通功率采用相同窗口；合并功率后取 log 与逐带取 log 是主要分频带对照。\n\n'
    report+='线性自身/模板控制共用对应联合模型在内折选定的正则强度，并未独立搜索各控制的最优正则；小幅超过控制不能作为已验证的稳健增益。特征噪声保留训练差分估计、相关矩阵收缩和合成 floor，不称为实测传感器标定。\n\n'
    report+='本轮载荷以固定训练 PCA 代理驱动回归，因而结论限定于这一拟合规则；它未联合学习一个与 PCA 低相关但可能对 fNIRS 有用的新驱动方向。负结果不能排除更合适载荷学习下的频带或空间收益。反向 EEG 只使用公共 clean 审计读出（中心缺失另评分隐藏段），不使用通用 runner 留下的候选原生 EEG 带噪指标作跨表示比较。\n\n'
    report+='完整 RF、10% 相对改善、每色团 NRMSE 退化≤0.05、至少两被试改善、配对 null 优势、合成恢复及必需模式完整性必须同时满足才保留。缺失的基线风险不以成功子集替代。当前 Gaussian MAP 不提供预测/状态可信区间、边际似然或 Student-t 校准；未取得 teacher/tokenizer 资格。\n\n'
    report+='完整输入的条件带噪重建可利用同源噪声交叉项，几乎零残差并不说明 clean 生理状态恢复；请与六状态真值恢复分开阅读。随机 EEG 特征点缺失 10/30% 和首局部通道丢失仅在共享合成 panel 检查，不能推断真实传感器丢失性能。\n\n'
    report+='## 图与证据\n\n'
    for name in figures:report+=f'![{name}](figures/{name}.png)\n\n'
    report+='逐行指标、失败、优化起点和内折选择见 analysis.json 及 CSV。原始配置、冻结源码、服务日志和资源记录在上级运行目录。首次准备的任务序列化失败保留在兄弟 v1 目录；没有从该次失败获得实测结论。所有图以 240 dpi PNG 嵌入 PDF，正文表格可搜索。未在 WPS 检查滚动/缩放。\n'
    (out/'REPORT.md').write_text(report)
    body=markdown.markdown(report,extensions=['tables']).replace('<img ','<img width="510" ')
    css='body{font-family:sans-serif;font-size:9pt;line-height:1.4}h1{font-size:20pt}h2{font-size:13pt}table{font-size:6.3pt;border-collapse:collapse}td,th{border:.5pt solid #cbd5df;padding:3pt}p{margin:6pt 0}'
    story=fitz.Story(body,user_css=css,archive=fitz.Archive(str(out)))
    def rectfn(n,filled):
        if n>70:raise RuntimeError('layout did not terminate')
        media=fitz.paper_rect('a4');return media,media+(36,32,-36,-32),None
    pdf=story.write_with_links(rectfn);pdf.save(out/'REPORT.pdf',garbage=4,deflate=True)
    audit=dict(pages=len(pdf),image_objects=sum(len(p.get_images()) for p in pdf),searchable_characters=sum(len(p.get_text()) for p in pdf),expected_figures=len(figures),bitmap_dpi=240,wps_checked=False)
    pdf.close();atlas=fitz.open()
    for name in figures:atlas.new_page(width=842,height=595).insert_image(fitz.Rect(25,25,817,570),filename=str(out/'figures'/f'{name}.png'),keep_proportion=True)
    atlas.save(out/'FIGURES.pdf',garbage=4,deflate=True);atlas.close()
    for name in figures:body=body.replace(f'figures/{name}.png','data:image/png;base64,'+base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode())
    (out/'REPORT.html').write_text('<html lang="zh-CN"><meta charset="utf-8"><style>'+css+'</style>'+body+'</html>')
    dump(out/'report_validation.json',audit)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--template-correction',type=Path)
    args=parser.parse_args()
    if read(args.run_dir/'manifest.json')['execution']!='completed':raise ValueError('terminal completed manifest required')
    render(args.run_dir,args.output_dir,analyze(args.run_dir,template_correction=args.template_correction))


if __name__=='__main__':main()
