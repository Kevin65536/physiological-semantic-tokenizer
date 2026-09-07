#!/usr/bin/env python3
"""Versioned Step5 stages; old Step1--4 and Step5A0 evidence stay immutable."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
from multiprocessing import get_context

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

import numpy as np
import scipy
import yaml
from scipy.special import logsumexp, poch
from scipy.stats import kstest, norm, spearmanr, t as student_t

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments import evaluate_step5a_inference_consistency as localization
from src.inference import t3a_balloon_joint_ssm as joint
from src.inference import t3a_balloon_robust_ssm as core

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml'
MEASURED_CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/step5b_v2.yaml'


def adaptive_particle_filter(y,cfg,axis,theta,particles,seed,resample_fraction=.5):
    """Bootstrap PF with correctly carried importance weights between resamples."""
    rng=np.random.default_rng(seed); m=cfg['model']; raw=localization.raw_parameters(cfg,axis,theta)
    z=rng.normal(size=(particles,6))*m['initial_state_std']
    weights=np.ones(particles)/particles; labels=np.arange(particles)
    ll,min_ess,resampling_count=0.,1.,0
    scales=np.array(m['observation_scale']); nu=m['student_nu']
    constant=np.log(poch(nu/2,.5))-.5*np.log(nu*np.pi)-np.log(scales)
    for t,row in enumerate(y):
        if t:
            if 1/np.sum(weights**2)<resample_fraction*particles:
                cdf=np.cumsum(weights); cdf[-1]=1.
                ix=np.searchsorted(cdf,(rng.random()+np.arange(particles))/particles)
                z=z[ix]; labels=labels[ix]; weights=np.ones(particles)/particles
                resampling_count+=1
            z=localization.independent_transition(z,raw,cfg)+rng.normal(size=z.shape)*np.array(m['process_std'])*np.sqrt(m['dt'])
        available=np.isfinite(row)
        residual=(row[available]-localization.clean_from_z(z)[:,available])/scales[available]
        log_density=np.sum(constant[available]-(nu+1)/2*np.log1p(residual**2/nu),axis=1)
        with np.errstate(divide='ignore'):
            log_weights=np.log(weights)+log_density
        increment=logsumexp(log_weights)
        ll+=increment; weights=np.exp(log_weights-increment)
        min_ess=min(min_ess,1/(particles*np.sum(weights**2)))
    return dict(parameter_log_likelihood=float(ll),minimum_ess_fraction=float(min_ess),
                unique_ancestor_fraction=len(np.unique(labels))/particles,resampling_count=resampling_count)


def joint_grid(y,cfg,axis,grid,order=None,smooth=False):
    order=order or cfg['inference']['quadrature_order']
    scores=[]; results=[]
    for value in grid:
        p,c=localization.model(cfg,axis,float(value))
        if smooth:
            result=joint.smooth_balloon_joint(y,p,config=c,quadrature_order=order)
            results.append(result); scores.append(result.parameter_log_likelihood)
        else:
            scores.append(joint.parameter_log_likelihood(y,p,config=c,quadrature_order=order))
    return np.array(scores),results


def reference_job(cfg,axis,level,run,index,theta,y):
    seed=cfg['seed']+10_000_000+list(cfg['axes']).index(axis)*1_000_000+level*100_000+run*1000+index
    return adaptive_particle_filter(y,cfg,axis,theta,cfg['reference']['particles'][level],seed,
                                    cfg['reference']['resample_ess_fraction'])


def write_json(path,value):
    localization.write_json(Path(path),value)


def reference_diagnostic(cfg,run_dir,workers):
    """Development reference check on the already retained localization cases."""
    run_dir=Path(run_dir)
    source=ROOT/cfg['reference']['localization_run']
    ref=cfg['reference']; R=ref['independent_runs']; N=ref['grid_points']
    rows={}; payload={}; logs={a:np.zeros((2,R,N)) for a in cfg['axes']}
    minima={a:dict(ess=1.,ancestry=1.) for a in cfg['axes']}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={}
        for axis in cfg['axes']:
            old=json.loads((source/f'case_{axis}_00.json').read_text())
            arrays=np.load(source/f'case_{axis}_00.npz')
            y=arrays['matched_model_calibration_observations'][:ref['steps']]
            grid=np.linspace(*cfg['axes'][axis]['bounds'],N)
            payload[axis]=(old,y,grid)
            for level in range(2):
                for r in range(R):
                    for j,theta in enumerate(grid):
                        job=pool.submit(reference_job,cfg,axis,level,r,j,theta,y)
                        jobs[job]=(axis,level,r,j)
        completed=0
        for job in as_completed(jobs):
            axis,level,r,j=jobs[job]; out=job.result(); logs[axis][level,r,j]=out['parameter_log_likelihood']
            minima[axis]['ess']=min(minima[axis]['ess'],out['minimum_ess_fraction'])
            minima[axis]['ancestry']=min(minima[axis]['ancestry'],out['unique_ancestor_fraction'])
            completed+=1
            if completed%100==0:print(f'reference {completed}/{len(jobs)}',flush=True)
    for axis,(old,y,grid) in payload.items():
        ll=logsumexp(logs[axis],axis=1)-np.log(R)
        cdfs=[localization.posterior_grid(cfg,axis,grid,l)['cdf'] for l in ll]
        split=[localization.posterior_grid(cfg,axis,grid,logsumexp(l,axis=0)-np.log(len(l)))['cdf'] for l in (logs[axis][1,:R//2],logs[axis][1,R//2:])]
        normalized=np.exp(logs[axis][1]-logs[axis][1].max(axis=0))
        se=normalized.std(axis=0,ddof=1)/np.sqrt(R)/normalized.mean(axis=0)
        coarse=localization.posterior_grid(cfg,axis,grid[::2],ll[1,::2])['cdf']
        budget_diff=float(np.max(abs(cdfs[0]-cdfs[1])))
        split_diff=float(np.max(abs(split[0]-split[1])))
        grid_diff=float(np.max(abs(cdfs[1]-np.interp(grid,grid[::2],coarse))))
        order=cfg['inference']['quadrature_order']
        jll,_=joint_grid(y,cfg,axis,grid,order)
        refined,_=joint_grid(y,cfg,axis,grid,cfg['inference']['quadrature_check_order'])
        jc=localization.posterior_grid(cfg,axis,grid,jll)['cdf']
        rc=localization.posterior_grid(cfg,axis,grid,refined)['cdf']
        oldgrid=np.array(old['nonlinear_reference']['grid'])
        oldc=np.interp(grid,oldgrid,localization.posterior_grid(cfg,axis,oldgrid,np.array(old['nonlinear_reference']['predictive_score']))['cdf'])
        likelihood_pass=bool(max(se)<=ref['max_log_likelihood_se'] and max(budget_diff,split_diff,grid_diff)<=ref['max_cdf_difference'] and minima[axis]['ess']>=ref['minimum_ess_fraction'])
        path_pass=bool(minima[axis]['ancestry']>=ref['minimum_unique_ancestor_fraction'])
        rows[axis]=dict(reference_precision_pass=likelihood_pass and path_pass,
                       parameter_likelihood_precision_pass=likelihood_pass,smoothing_path_diversity_pass=path_pass,
                       maximum_log_likelihood_se=float(max(se)),
                       minimum_ess_fraction=minima[axis]['ess'],minimum_unique_ancestor_fraction=minima[axis]['ancestry'],
                       particle_budget_cdf_difference=budget_diff,split_run_cdf_difference=split_diff,
                       reference_grid_cdf_difference=grid_diff,joint_to_reference_cdf_difference=float(max(abs(jc-cdfs[1]))),
                       old_score_to_reference_cdf_difference=float(max(abs(oldc-cdfs[1]))),
                       quadrature_cdf_difference=float(max(abs(jc-rc))),quadrature_log_likelihood_max_abs=float(max(abs(jll-refined))),
                       grid=grid,independent_reference_log_likelihoods=logs[axis],reference_parameter_log_likelihood=ll[1],joint_parameter_log_likelihood=jll)
        write_json(run_dir/f'reference_{axis}.json',rows[axis])
    result=dict(stage='a0_reference',axes={a:{k:v for k,v in row.items() if k not in {'grid','independent_reference_log_likelihoods','reference_parameter_log_likelihood','joint_parameter_log_likelihood'}} for a,row in rows.items()})
    write_json(run_dir/'summary.json',result)
    return result


def validate_config(cfg):
    if cfg['schema']!='step5_v1' or cfg['scope']!='staged_synthetic_then_nonprotected_development':
        raise ValueError('invalid Step5 scope or schema')
    if cfg['output_root']!='experiments/runs/physiology_semantic_tokenizer/step5':
        raise ValueError('Step5 output root drift')
    if set(cfg['axes'])!={'G','W','Z'} or cfg['replicates_per_axis']!=60:
        raise ValueError('registered full calibration requires 60 independent replicates per axis')
    for axis in cfg['axes']:
        spec=cfg['axes'][axis]
        if spec['prior']!=('uniform' if axis=='Z' else 'truncated_normal'):
            raise ValueError('prior coordinate/family drift')
        lo,hi=spec['bounds']
        if lo>=hi or (axis=='Z' and not 0<lo<hi<1):raise ValueError('invalid parameter support')
        law=localization.prior(cfg,axis)
        if not np.isfinite(law.mean()) or law.std()<=0:
            raise ValueError('invalid prior')
    for section in ('inference','oracle'):
        for key in ('grid_points','maximum_grid_points'):
            if cfg[section][key]<5 or cfg[section][key]%2!=1:
                raise ValueError('nested grids require odd sizes >= 5')
    p,c=localization.model(cfg,'G',0.)
    p.validate(); c.validate()
    if cfg['measured']['subjects']!=[f'subject_{i:02d}' for i in range(1,19)]:
        raise ValueError('Step5 measured scope is exactly subjects 01--18')
    if cfg['measured']['sessions']!=['session_01','session_03','session_05']:
        raise ValueError('wrong session scope')
    if cfg['measured']['replication_subjects'] or cfg['measured']['protected_subjects']!=[f'subject_{i:02d}' for i in range(24,30)]:
        raise ValueError('this version cannot open replication or protected arrays')
    if cfg['measured']['heldout_trial_positions']!=[4,9]:
        raise ValueError('fixed per-session heldout trial identities drifted')
    if cfg['inference']['method']!='joint_student_t_gaussian_moment_filter_rts':
        raise ValueError('invalid inference owner')
    for key in ('quadrature_order','quadrature_check_order'):
        joint.normal_quadrature(cfg['inference'][key],1)


def load_config(path=CONFIG):
    cfg=yaml.safe_load(Path(path).read_text()); validate_config(cfg); return cfg


def cluster_summary(values,cfg,seed=0):
    values=np.array(values,dtype=float)
    if values.ndim!=1 or len(values)<1 or not np.isfinite(values).all():
        raise ValueError('cluster summary requires finite independent-unit values')
    rng=np.random.default_rng(seed)
    boot=values[rng.integers(0,len(values),(cfg['inference']['bootstrap_repetitions'],len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]),independent_units=len(values))


def targets_and_moments(generated,result):
    truth=np.column_stack((generated['states'][:,0],generated['clean']))
    mean=np.column_stack((result.state_mean[:,0],result.trajectory_mean))
    variance=np.column_stack((result.state_covariance[:,0,0],result.state_posterior_variance))
    return truth,mean,variance


def truth_metrics(truth,mean,variance):
    result={}
    for j,name in enumerate(localization.TARGETS):
        error=mean[:,j]-truth[:,j]; sd=np.sqrt(np.maximum(variance[:,j],0))
        corr=np.corrcoef(truth[:,j],mean[:,j])[0,1]
        result[name]=dict(rmse=float(np.sqrt(np.mean(error**2))),
                         nrmse=float(np.sqrt(np.mean(error**2))/max(np.std(truth[:,j]),1e-12)),
                         correlation=float(corr) if np.isfinite(corr) else 0.,
                         coverage90=float(np.mean(abs(error)<=norm.ppf(.95)*sd)),
                         coverage95=float(np.mean(abs(error)<=norm.ppf(.975)*sd)),
                         mean_state_posterior_variance=float(np.mean(variance[:,j])))
    return result


def refined_joint_grid(y,cfg,axis,smooth=False):
    n=cfg['inference']['grid_points']; maximum=cfg['inference']['maximum_grid_points']
    grid=np.linspace(*cfg['axes'][axis]['bounds'],n)
    ll,states=joint_grid(y,cfg,axis,grid,smooth=smooth)
    while True:
        post=localization.posterior_grid(cfg,axis,grid,ll)
        coarse=localization.posterior_grid(cfg,axis,grid[::2],ll[::2])
        diff=float(max(abs(post['cdf']-np.interp(grid,grid[::2],coarse['cdf']))))
        if diff<=cfg['calibration']['grid_max_cdf_difference'] or len(grid)>=maximum:break
        fine=np.linspace(grid[0],grid[-1],2*len(grid)-1)
        added,added_states=joint_grid(y,cfg,axis,fine[1::2],smooth=smooth)
        values=np.empty(len(fine)); values[::2]=ll; values[1::2]=added
        if smooth:
            combined=[None]*len(fine); combined[::2]=states; combined[1::2]=added_states; states=combined
        grid,ll=fine,values
    return grid,ll,states,diff


def calibration_case(cfg,axis,replicate):
    seed=cfg['seed']+list(cfg['axes']).index(axis)*1_000_000+replicate*1000
    theta=float(localization.prior(cfg,axis).rvs(random_state=np.random.default_rng(seed)))
    generated=localization.generate_matched(cfg,axis,theta,seed+1)
    y=generated['observations']
    grid,ll,_,resolution=refined_joint_grid(y,cfg,axis)
    joint_posterior=localization.posterior_summary(cfg,axis,grid,ll,theta)
    old_score=localization.fit_score_grid(y,cfg,axis,grid)
    old_posterior=localization.posterior_summary(cfg,axis,grid,old_score,theta)
    p,c=localization.model(cfg,axis,theta)
    state=joint.smooth_balloon_joint(y,p,config=c,quadrature_order=cfg['inference']['quadrature_order'])
    truth,mean,variance=targets_and_moments(generated,state)
    result=dict(axis=axis,replicate=replicate,seed=seed,truth=theta,grid=grid,
                parameter_log_likelihood=ll,predictive_score=old_score,
                joint_posterior=joint_posterior,legacy_generalized_posterior=old_posterior,
                grid_cdf_difference=resolution,
                conditional_true_parameter_state_metrics=truth_metrics(truth,mean,variance),
                physical_checks=dict(state.physical_checks))
    if replicate in cfg['calibration']['quadrature_check_replicates']:
        check,_=joint_grid(y,cfg,axis,grid,cfg['inference']['quadrature_check_order'])
        pc=localization.posterior_grid(cfg,axis,grid,check)['cdf']
        jc=localization.posterior_grid(cfg,axis,grid,ll)['cdf']
        result['quadrature_cdf_difference']=float(np.max(abs(pc-jc)))
    oracle_cfg=copy.deepcopy(cfg)
    oracle_cfg['inference']['grid_points']=cfg['oracle']['grid_points']
    while True:
        oracle=localization.oracle_case(oracle_cfg,axis,theta,seed+2)
        if oracle['quadrature_pass'] or oracle_cfg['inference']['grid_points']>=cfg['oracle']['maximum_grid_points']:break
        oracle_cfg['inference']['grid_points']=2*oracle_cfg['inference']['grid_points']-1
    result['oracle']=oracle
    arrays=dict(truth_states=generated['states'],truth_clean=generated['clean'],observations=y,
                conditional_state_mean=state.state_mean,conditional_state_covariance=state.state_covariance,
                conditional_clean_mean=state.trajectory_mean,conditional_clean_variance=state.state_posterior_variance)
    return result,arrays


def calibration_summary(cfg,rows,reference):
    from scipy.stats import binomtest
    summary=dict(stage='a0',execution='completed',independent_cases=len(rows),axes={})
    for axis in cfg['axes']:
        cases=[r for r in rows if r['axis']==axis]; n=len(cases); values={}
        for method in ('joint_posterior','legacy_generalized_posterior','oracle'):
            estimates=[r[method] for r in cases]
            ranks=np.array([r['rank_u'] for r in estimates])
            covered=sum(r['covered95'] for r in estimates)
            ci=binomtest(covered,n).proportion_ci(confidence_level=cfg['calibration']['parameter_coverage_confidence'])
            ks=kstest(ranks,'uniform')
            bias=[r[method]['mean']-r['truth'] for r in cases]
            values[method]=dict(rank_ks_statistic=float(ks.statistic),rank_ks_p=float(ks.pvalue),
                                rank_mean=float(ranks.mean()),coverage95=covered/n,coverage95_ci=[ci.low,ci.high],
                                posterior_mean_bias=cluster_summary(bias,cfg,cfg['seed']),
                                mean_interval_width_fraction=float(np.mean([r['width_fraction'] for r in estimates])),
                                mean_boundary_mass=float(np.mean([r['boundary_mass'] for r in estimates])),
                                calibration_screen_pass=bool(ks.pvalue>=cfg['calibration']['ks_minimum_p'] and ci.low<=.95<=ci.high))
        values['maximum_grid_cdf_difference']=max(r['grid_cdf_difference'] for r in cases)
        values['maximum_quadrature_cdf_difference']=max(r.get('quadrature_cdf_difference',0) for r in cases)
        values['oracle_resolution_passes']=sum(r['oracle']['quadrature_pass'] for r in cases)
        values['state_metrics']={}
        for name in localization.TARGETS:
            values['state_metrics'][name]={metric:cluster_summary([r['conditional_true_parameter_state_metrics'][name][metric] for r in cases],cfg,cfg['seed']+20)
                                          for metric in ('rmse','nrmse','correlation','coverage90','coverage95')}
        reference_row=reference['axes'][axis]
        values['reference']=reference_row
        values['numerical_inference_screen_pass']=bool(
            values['maximum_grid_cdf_difference']<=cfg['calibration']['grid_max_cdf_difference'] and
            values['maximum_quadrature_cdf_difference']<=cfg['inference']['quadrature_max_cdf_difference'] and
            values['oracle_resolution_passes']==n and reference_row['parameter_likelihood_precision_pass'] and
            reference_row['joint_to_reference_cdf_difference']<=cfg['reference']['maximum_joint_reference_cdf_difference'])
        summary['axes'][axis]=values
    summary['numerical_inference_screen_pass']=all(v['numerical_inference_screen_pass'] for v in summary['axes'].values())
    summary['parameter_calibration_screen_pass']=all(v['joint_posterior']['calibration_screen_pass'] for v in summary['axes'].values())
    summary['teacher_qualification']='NOT_EVALUATED_UNTIL_STEP5A1'
    return summary


def phase_report(cfg,summary):
    if summary['stage']!='a0':raise ValueError('stage report not implemented')
    lines=['# Step5A0 阶段报告：联合观测推断修正与同模型校准','',
           '本阶段保留六状态共享 Balloon 和 GWZ 坐标；新增联合 Student-t 数值积分及 Gaussian moment matching，原 Step 1–4 和初始 Step5A0 代码/证据保持不变。',
           '似然仍受 EKF 转移与 Gaussian filtering closure 的近似影响，故下述校准属于数值与统计验证，不能仅凭实现形式宣称精确后验。','',
           f"按冻结合同运行 G/W/Z 各 {cfg['replicates_per_axis']} 个独立重复，共 {summary['independent_cases']} 例；每例 40 秒，初始分布、离散转移、六维过程扩散及 t5 观测噪声与拟合模型一致。",'',
           '| 方向 | 方法 | rank KS p | 参数95%覆盖 | 覆盖95%二项区间 | 后验均值偏差 |','|---|---|---:|---:|---|---:|']
    for axis,values in summary['axes'].items():
        for method in ('oracle','legacy_generalized_posterior','joint_posterior'):
            r=values[method]
            lines.append(f"| {axis} | {method} | {r['rank_ks_p']:.6f} | {r['coverage95']:.4f} | [{r['coverage95_ci'][0]:.4f}, {r['coverage95_ci'][1]:.4f}] | {r['posterior_mean_bias']['mean']:.5f} |")
    lines+=['','| 方向 | 联合/参考CDF差 | 旧评分/参考CDF差 | GH积分CDF差 | 参数网格CDF差 | Oracle分辨率通过 |','|---|---:|---:|---:|---:|---:|']
    for axis,v in summary['axes'].items():
        r=v['reference']
        lines.append(f"| {axis} | {r['joint_to_reference_cdf_difference']:.6f} | {r['old_score_to_reference_cdf_difference']:.6f} | {v['maximum_quadrature_cdf_difference']:.6f} | {v['maximum_grid_cdf_difference']:.6f} | {v['oracle_resolution_passes']}/60 |")
    lines+=['','| 方向 | r 95%覆盖 | clean EEG | clean HbO | clean HbR |','|---|---:|---:|---:|---:|']
    for axis,v in summary['axes'].items():
        lines.append(f'| {axis} | '+' | '.join(f"{v['state_metrics'][n]['coverage95']['mean']:.4f}" for n in localization.TARGETS)+' |')
    lines+=['',f"数值推断检查：**{summary['numerical_inference_screen_pass']}**；参数校准检查：**{summary['parameter_calibration_screen_pass']}**。",
            '参数 coverage 是独立参数重复的二项区间；状态 coverage 先按 replicate 汇总再 bootstrap，未把自相关时间点当独立重复。参数级结果不会自动授予或剥夺状态 teacher 资格。',
            'W 的粒子似然精度与路径多样性分开判读：本阶段只使用其似然参考，不使用退化祖先路径提供状态平滑真值。旧版综合参考门的失败记录保留。',
            '此处状态区间条件于真实参数；参数传播、U0 敏感性、跨参数稳定性以及双向共享信息 null 在 Step5A1 判定。未读取任何 measured/protected 数组。','']
    return '\n'.join(lines)


def start_stage(cfg,stage,run_dir):
    run_dir=Path(run_dir).resolve()
    if run_dir.parent!=(ROOT/cfg['output_root']).resolve():raise ValueError('fresh direct child of Step5 root required')
    run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    sources=[Path(__file__),ROOT/'src/inference/t3a_balloon_joint_ssm.py',ROOT/'src/inference/t3a_balloon_robust_ssm.py',ROOT/'experiments/evaluate_step5a_inference_consistency.py']
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (run_dir/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (run_dir/'inference_snapshot.py').write_bytes(sources[1].read_bytes())
    manifest=dict(schema=cfg['schema'],stage=stage,execution='running',started_at=datetime.now(timezone.utc).isoformat(),
                  config_sha256=hashlib.sha256((run_dir/'resolved_config.yaml').read_bytes()).hexdigest(),
                  source_sha256=source_hashes,numpy=np.__version__,scipy=scipy.__version__,measured_data_read=False)
    write_json(run_dir/'manifest.json',manifest)
    return run_dir,manifest


def run_calibration(cfg,run_dir,reference_run,workers):
    validate_config(cfg)
    refdir=Path(reference_run).resolve(); reference=json.loads((refdir/'summary.json').read_text())
    if reference['stage']!='a0_reference':raise ValueError('wrong reference stage')
    reference_config=yaml.safe_load((refdir/'resolved_config.yaml').read_text())
    if any(reference_config[key]!=cfg[key] for key in ('model','axes')):
        raise ValueError('reference model or parameter prior differs from calibration')
    if not all(v['parameter_likelihood_precision_pass'] for v in reference['axes'].values()):
        raise ValueError('parameter likelihood reference unresolved')
    run_dir,manifest=start_stage(cfg,'a0',run_dir); started=time.time(); rows=[]
    manifest['reference_run']=str(refdir.relative_to(ROOT))
    manifest['reference_summary_sha256']=hashlib.sha256((refdir/'summary.json').read_bytes()).hexdigest()
    try:
        checks=dict(parameterization=localization.parameterization_check(cfg),linear_kalman=localization.linear_gaussian_check(cfg))
        write_json(run_dir/'implementation_checks.json',checks)
        if not all(v['passed'] for v in checks.values()):raise RuntimeError('implementation checks failed')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs={pool.submit(calibration_case,cfg,a,r):(a,r) for a in cfg['axes'] for r in range(cfg['replicates_per_axis'])}
            for job in as_completed(jobs):
                row,arrays=job.result(); rows.append(row)
                stem=f"case_{row['axis']}_{row['replicate']:02d}"
                write_json(run_dir/f'{stem}.json',row); np.savez_compressed(run_dir/f'{stem}.npz',**arrays)
                print(f"a0 {len(rows)}/{len(jobs)} {stem}",flush=True)
        rows.sort(key=lambda r:(r['axis'],r['replicate']))
        summary=calibration_summary(cfg,rows,reference)
        write_json(run_dir/'summary.json',summary)
        (run_dir/'summary.md').write_text(phase_report(cfg,summary))
        manifest.update(execution='completed',completed_cases=len(rows),elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(execution='failed',completed_cases=len(rows),error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.time()-started)
        raise
    finally:write_json(run_dir/'manifest.json',manifest)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=['reference','a0','a1','review','b','report'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path)
    parser.add_argument('--calibration-run',type=Path)
    parser.add_argument('--teacher-run',type=Path)
    parser.add_argument('--measured-config',type=Path,default=MEASURED_CONFIG)
    parser.add_argument('--reuse-run',type=Path)
    parser.add_argument('--workers',type=int,default=None)
    args=parser.parse_args(); cfg=load_config(args.config)
    workers=args.workers or cfg['workers']
    if args.stage=='reference':
        run_dir,manifest=start_stage(cfg,'a0_reference',args.run_dir);started=time.time()
        try:
            reference_diagnostic(cfg,run_dir,workers)
            manifest.update(execution='completed',workers=workers)
        except Exception as exc:
            manifest.update(execution='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            manifest['elapsed_seconds']=time.time()-started;write_json(run_dir/'manifest.json',manifest)
    elif args.stage=='a0':
        if not args.reference_run:parser.error('--reference-run is required for a0')
        run_calibration(cfg,args.run_dir,args.reference_run,workers)
    elif args.stage=='a1':
        if not args.calibration_run:parser.error('--calibration-run is required for a1')
        run_teacher(cfg,args.run_dir,args.calibration_run,workers)
    elif args.stage=='review':
        if not args.teacher_run:parser.error('--teacher-run is required for review')
        run_candidate_scope_review(cfg,args.run_dir,args.teacher_run)
    elif args.stage=='b':
        if not args.teacher_run:parser.error('--teacher-run is required for b')
        run_measured(args.measured_config,args.run_dir,args.teacher_run,args.workers,args.reuse_run)
    else:
        report_step5(args.run_dir)
    return 0



def case_seed(cfg,axis,replicate,stream):
    axis_id={'G':0,'W':1,'Z':2,'R':3}[axis]
    return int(np.random.SeedSequence([cfg['seed'],51,axis_id,replicate,stream]).generate_state(1,dtype=np.uint64)[0])


def parameter_nodes(cfg,axis,training_fit,count):
    from scipy.special import roots_legendre
    if training_fit is None:return np.array([0.]),np.array([1.])
    grid=np.array(training_fit['grid']); likelihood=np.array(training_fit['parameter_log_likelihood'])
    cdf=localization.posterior_grid(cfg,axis,grid,likelihood)['cdf']
    q,w=roots_legendre(count); q=(q+1)/2; w=w/2
    return np.interp(q,cdf,grid),w


def mixture_state(y,cfg,axis,training_fit,count=None):
    count=count or cfg['teacher']['frozen_training_parameter_mixture_points']
    nodes,weights=parameter_nodes(cfg,axis,training_fit,count)
    results=[]
    for value in nodes:
        p,c=localization.model(cfg,axis if training_fit is not None else 'G',float(value))
        results.append(joint.smooth_balloon_joint(y,p,config=c,quadrature_order=cfg['inference']['quadrature_order']))
    state_means=np.array([r.state_mean for r in results])
    clean_means=np.array([r.trajectory_mean for r in results])
    state_variances=np.array([np.diagonal(r.state_covariance,axis1=1,axis2=2) for r in results])
    clean_variances=np.array([r.state_posterior_variance for r in results])
    state_mean=np.einsum('n,ntk->tk',weights,state_means)
    clean_mean=np.einsum('n,ntk->tk',weights,clean_means)
    conditional_state_var=np.einsum('n,ntk->tk',weights,state_variances)
    conditional_clean_var=np.einsum('n,ntk->tk',weights,clean_variances)
    parameter_state_var=np.einsum('n,ntk->tk',weights,(state_means-state_mean)**2)
    parameter_clean_var=np.einsum('n,ntk->tk',weights,(clean_means-clean_mean)**2)
    checks=('finite','positive_fvpq','oxygen_extraction_in_unit_interval','absolute_hb_nonnegative','hbr_not_above_hbt')
    return dict(state_mean=state_mean,clean_mean=clean_mean,
                conditional_state_variance=conditional_state_var,parameter_state_variance=parameter_state_var,
                conditional_clean_variance=conditional_clean_var,parameter_clean_variance=parameter_clean_var,
                state_variance=conditional_state_var+parameter_state_var,clean_variance=conditional_clean_var+parameter_clean_var,
                component_clean_means=clean_means,component_clean_variances=clean_variances,
                component_driver_means=state_means[:,:,0],parameter_nodes=nodes,parameter_weights=weights,
                physical_pass=all(all(r.physical_checks[k] for k in checks) for r in results))


def mixture_truth_metrics(generated,estimate,selector=None):
    truth=np.column_stack((generated['states'][:,0],generated['clean']))
    mean=np.column_stack((estimate['state_mean'][:,0],estimate['clean_mean']))
    var=np.column_stack((estimate['state_variance'][:,0],estimate['clean_variance']))
    if selector is not None:truth,mean,var=truth[selector],mean[selector],var[selector]
    return truth_metrics(truth,mean,var)


def masked_input(y,mask_name,center_steps=16):
    visible=np.array(y,copy=True)
    columns=[0] if mask_name.endswith('EEG') else [1,2]
    time_mask=np.ones(len(y),dtype=bool)
    if mask_name.startswith('center'):
        if not 0<center_steps<len(y):raise ValueError('invalid center gap')
        time_mask[:]=False; start=(len(y)-center_steps)//2; time_mask[start:start+center_steps]=True
    visible[np.ix_(time_mask,columns)]=np.nan
    return visible,time_mask,columns


def predictive_log_score(truth,estimate,time_mask,columns,cfg):
    nodes,logweights=joint.normal_quadrature(cfg['teacher']['predictive_quadrature_order'],1)
    means=estimate['component_clean_means'][:,time_mask][:,:,columns]
    variance=estimate['component_clean_variances'][:,time_mask][:,:,columns]
    latent=means[...,None]+np.sqrt(np.maximum(variance,0))[...,None]*nodes[:,0]
    scales=np.array(cfg['model']['observation_scale'])[columns]
    nu=cfg['model']['student_nu']
    residual=(np.asarray(truth)[time_mask][:,columns][None,...,None]-latent)/scales[None,None,:,None]
    constant=np.log(poch(nu/2,.5))-.5*np.log(nu*np.pi)-np.log(scales)
    density=constant[None,None,:,None]-(nu+1)/2*np.log1p(residual**2/nu)
    conditional=logsumexp(density+logweights,axis=-1)
    log_score=logsumexp(conditional+np.log(estimate['parameter_weights'])[:,None,None],axis=0)
    return float(np.mean(log_score))


def cross_parameter_stability(estimate):
    mean=estimate['state_mean'][:,0]; weights=estimate['parameter_weights']
    curves=estimate['component_driver_means']
    corr=np.array([np.corrcoef(mean,row)[0,1] for row in curves])
    nrmse=np.sqrt(np.mean((curves-mean)**2,axis=1))/max(np.std(mean),1e-12)
    def weighted_quantile(x,q):
        order=np.argsort(x); return float(np.interp(q,np.cumsum(weights[order]),x[order]))
    return dict(driver_correlation_q05=weighted_quantile(corr,.05),driver_nrmse_q95=weighted_quantile(nrmse,.95))


def sensitivity_u0(cfg,y,generated,baseline):
    base=mixture_truth_metrics(generated,baseline)
    rows=[]
    for axis in cfg['axes']:
        for q in (.05,.95):
            theta=float(localization.prior(cfg,axis).ppf(q));p,c=localization.model(cfg,axis,theta)
            r=joint.smooth_balloon_joint(y,p,config=c,quadrature_order=cfg['inference']['quadrature_order'])
            mean=np.column_stack((r.state_mean[:,0],r.trajectory_mean))
            var=np.column_stack((r.state_covariance[:,0,0],r.state_posterior_variance))
            truth=np.column_stack((generated['states'][:,0],generated['clean']))
            m=truth_metrics(truth,mean,var)
            driver=baseline['state_mean'][:,0]
            rows.append(dict(axis=axis,prior_quantile=q,theta=theta,
                             driver_correlation=float(np.corrcoef(driver,r.state_mean[:,0])[0,1]),
                             driver_nrmse=float(np.sqrt(np.mean((driver-r.state_mean[:,0])**2))/max(np.std(driver),1e-12)),
                             maximum_clean_nrmse_increase=max(m[t]['nrmse']-base[t]['nrmse'] for t in localization.TARGETS[1:])))
    return rows


def teacher_case(cfg,axis,replicate,calibration_dir):
    if axis=='R':
        truth=0.;fit=None
    else:
        original=json.loads((Path(calibration_dir)/f'case_{axis}_{replicate:02d}.json').read_text())
        truth=original['truth'];fit={k:original[k] for k in ('grid','parameter_log_likelihood')}
    generator_axis=axis if axis!='R' else 'G'
    row=dict(axis=axis,replicate=replicate,truth=truth,laws={}); payload={}
    for law in ('matched_model_calibration','misspecification_stress_test'):
        offset=0 if law=='matched_model_calibration' else 100
        def generate(stream):
            seed=case_seed(cfg,axis,replicate,offset+stream)
            if law=='matched_model_calibration':return localization.generate_matched(cfg,generator_axis,truth,seed)
            return localization.step4_truth(localization.raw_parameters(cfg,generator_axis,truth),seed,localization.stress_contract(cfg))
        generated=generate(1); y=generated['observations']; donor=generate(2)['observations']
        template=np.mean([generate(10+j)['observations'] for j in range(cfg['teacher']['task_template_training_trials'])],axis=0)
        if law=='misspecification_stress_test' and axis!='R':
            training=generate(3)
            grid,ll,_,grid_difference=refined_joint_grid(training['observations'],cfg,axis)
            law_fit=dict(grid=grid,parameter_log_likelihood=ll)
        else:
            law_fit=fit;grid_difference=0.
        candidates={'U0_FIXED':None}
        if axis!='R':candidates[f'U1_{axis}']=law_fit
        out={}
        for candidate,training_fit in candidates.items():
            full=mixture_state(y,cfg,generator_axis,training_fit)
            result=dict(full_truth=mixture_truth_metrics(generated,full),physical_pass=full['physical_pass'],
                        stability=cross_parameter_stability(full),masks={},training_grid_cdf_difference=grid_difference)
            if training_fit is not None:
                refined=mixture_state(y,cfg,axis,training_fit,cfg['teacher']['parameter_mixture_check_points'])
                result['mixture_mean_difference_in_noise_units']=float(np.max(abs(full['clean_mean']-refined['clean_mean'])/np.array(cfg['model']['observation_scale'])))
                result['mixture_relative_variance_difference']=float(np.max(abs(full['clean_variance']-refined['clean_variance'])/np.maximum(refined['clean_variance'],1e-12)))
            else:
                result['sensitivity']=sensitivity_u0(cfg,y,generated,full)
            result['variance_decomposition']={name:dict(
                state=float(full['conditional_clean_variance'][:,j].mean()),
                parameter=float(full['parameter_clean_variance'][:,j].mean()),
                observation=float(np.square(cfg['model']['observation_scale'][j])*cfg['model']['student_nu']/(cfg['model']['student_nu']-2))) for j,name in enumerate(('EEG','HbO','HbR'))}
            for mask_name in cfg['teacher']['masks']:
                visible,time_mask,columns=masked_input(y,mask_name,cfg['inference']['mask_center_steps'])
                other=[j for j in range(3) if j not in columns]
                estimate=mixture_state(visible,cfg,generator_axis,training_fit)
                score=predictive_log_score(y,estimate,time_mask,columns,cfg)
                mask_result=dict(joint_score=score,truth_metrics=mixture_truth_metrics(generated,estimate,time_mask),controls={})
                basevariance=np.column_stack((full['state_variance'][:,0],full['clean_variance']))[time_mask]
                maskedvariance=np.column_stack((estimate['state_variance'][:,0],estimate['clean_variance']))[time_mask]
                mask_result['paired_variance_change']=np.mean(maskedvariance-basevariance,axis=0)
                if mask_name in cfg['teacher']['primary_masks']:
                    for control in cfg['teacher']['controls']:
                        null_input=visible.copy()
                        if control=='own_history':null_input[:,other]=np.nan
                        elif control=='own_history_and_task':
                            null_input-=template;null_input[:,other]=np.nan
                        elif control=='independent_pairing':null_input[:,other]=donor[:,other]
                        elif control=='circular_shift':null_input[:,other]=np.roll(y[:,other],int(len(y)*cfg['teacher']['circular_shift_fraction']),axis=0)
                        else:raise ValueError('unknown control')
                        null=mixture_state(null_input,cfg,generator_axis,training_fit)
                        if control=='own_history_and_task':
                            null['clean_mean']+=template
                            null['component_clean_means']+=template[None]
                        control_score=predictive_log_score(y,null,time_mask,columns,cfg)
                        mask_result['controls'][control]=dict(score=control_score,joint_increment=score-control_score)
                result['masks'][mask_name]=mask_result
                payload[f'{law}_{candidate}_{mask_name}_mean']=estimate['clean_mean']
                payload[f'{law}_{candidate}_{mask_name}_variance']=estimate['clean_variance']
            out[candidate]=result
            payload[f'{law}_{candidate}_state_mean']=full['state_mean']
            payload[f'{law}_{candidate}_state_variance']=full['state_variance']
            payload[f'{law}_{candidate}_clean_mean']=full['clean_mean']
            payload[f'{law}_{candidate}_clean_variance']=full['clean_variance']
        row['laws'][law]=out
        for key in ('observations','states','clean'):payload[f'{law}_{key}']=generated[key]
    return row,payload


def multi_parameter_model(cfg,values):
    g,w,zeta=values
    p,c=localization.model(cfg,'G',0.)
    reference=cfg['model']['reference']; gamma=reference['gamma']*np.exp(2*w)
    p=replace(p,fixed=replace(p.fixed,neurovascular_gain=reference['beta']*np.exp(g+2*w),gamma=gamma),
              free=replace(p.free,kappa=2*zeta*np.sqrt(gamma)))
    return p,c


def u3_chunk(cfg,y,coordinates):
    scores=[]
    for point in coordinates:
        p,c=multi_parameter_model(cfg,point)
        scores.append(joint.parameter_log_likelihood(y,p,config=c,quadrature_order=cfg['inference']['quadrature_order']))
    return scores


def u3_diagnostic(cfg,calibration_dir,run_dir,workers):
    from itertools import product
    results={}
    for axis,replicate in zip(('G','W','Z'),cfg['teacher']['u3_diagnostic_replicates']):
        record=json.loads((Path(calibration_dir)/f'case_{axis}_{replicate:02d}.json').read_text())
        y=np.load(Path(calibration_dir)/f'case_{axis}_{replicate:02d}.npz')['observations']
        levels=[]
        for n in (cfg['teacher']['u3_quadrature_points_per_axis'],cfg['teacher']['u3_refinement_points_per_axis']):
            grids=[np.linspace(*cfg['axes'][a]['bounds'],n) for a in ('G','W','Z')]
            points=np.array(list(product(*grids)))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                chunks=list(pool.map(u3_chunk,[cfg]*((len(points)+31)//32),[y]*((len(points)+31)//32),[points[j:j+32] for j in range(0,len(points),32)]))
            ll=np.concatenate(chunks).reshape(n,n,n)
            weight=np.ones((n,n,n))
            for j,a in enumerate(('G','W','Z')):
                spacing=grids[j][1]-grids[j][0]; q=np.ones(n)*spacing;q[[0,-1]]*=.5
                q*=localization.prior(cfg,a).pdf(grids[j])
                shape=[1,1,1];shape[j]=n;weight*=q.reshape(shape)
            weight*=np.exp(ll-np.max(ll));weight/=weight.sum()
            flat=weight.ravel();mean=flat@points;delta=points-mean
            covariance=(delta*flat[:,None]).T@delta
            marginal=[];cdf=[];boundary=[]
            for j,a in enumerate(('G','W','Z')):
                m=weight.sum(axis=tuple(k for k in range(3) if k!=j));marginal.append(m)
                # Integral nodes carry trapezoidal probability weights.
                midcdf=np.cumsum(m)-m/2;midcdf[0]=0.;midcdf[-1]=1.;cdf.append(midcdf)
                lo,hi=cfg['axes'][a]['bounds'];band=cfg['inference']['boundary_band_fraction']*(hi-lo)
                boundary.append(float(np.interp(lo+band,grids[j],midcdf)+1-np.interp(hi-band,grids[j],midcdf)))
            sd=np.sqrt(np.diag(covariance));corr=covariance/np.outer(sd,sd)
            levels.append(dict(grid_points_per_axis=n,grids=grids,parameter_log_likelihood=ll,
                               posterior_weights=weight,mean=mean,covariance=covariance,correlation=corr,
                               marginal_cdfs=cdf,boundary_mass=boundary,
                               effective_grid_points=float(1/np.sum(flat**2))))
            print(f'a1 U3 {axis}_{replicate:02d} {n}^3 completed',flush=True)
        coarse,fine=levels
        differences=[float(max(abs(fine['marginal_cdfs'][j]-np.interp(fine['grids'][j],coarse['grids'][j],coarse['marginal_cdfs'][j])))) for j in range(3)]
        row=dict(case=f'{axis}_{replicate:02d}',truth_axis=axis,truth=record['truth'],levels=levels,
                 marginal_grid_cdf_differences=differences,
                 numerical_status='RESOLVED_DIAGNOSTIC' if max(differences)<=cfg['reference']['max_cdf_difference'] else 'INCONCLUSIVE_GRID_RESOLUTION',
                 selection_eligible=False)
        write_json(Path(run_dir)/f'u3_{axis}_{replicate:02d}.json',row)
        results[f'{axis}_{replicate:02d}']={k:v for k,v in row.items() if k!='levels'}
        results[f'{axis}_{replicate:02d}']['posterior_correlation']=fine['correlation']
        results[f'{axis}_{replicate:02d}']['boundary_mass']=fine['boundary_mass']
    return results


def teacher_summary(cfg,rows,u3):
    failures=[r for r in rows if r.get('execution')=='failed']
    successes=[r for r in rows if r.get('execution')!='failed']
    summary=dict(stage='a1',execution='completed',independent_cases=len(rows),successful_cases=len(successes),
                 failed_cases=len(failures),case_failures=failures,candidates={},u3_diagnostic=u3)
    for candidate in cfg['teacher']['candidates']:
        axis='R' if candidate=='U0_FIXED' else candidate[-1]
        cases=[r for r in successes if r['axis']==axis]
        # U1 axes have separate parameter priors and independent registered
        # panels. A failed G/Z panel cannot invalidate all 60 complete W cases.
        # U0 sensitivity, by contrast, explicitly depends on every scenario.
        relevant_failures=failures if candidate=='U0_FIXED' else [r for r in failures if r['axis']==axis]
        result=dict(independent_cases=len(cases),laws={},checks={
            'complete_registered_cases':len(rows)==4*cfg['teacher']['matching_replicates']
                and len(cases)==cfg['teacher']['matching_replicates'] and not relevant_failures})
        if not cases:
            result.update(teacher_qualified=False,parameter_physiology_claim='NOT_GRANTED_BY_STATE_QUALIFICATION')
            summary['candidates'][candidate]=result
            continue
        for law in ('matched_model_calibration','misspecification_stress_test'):
            entries=[r['laws'][law][candidate] for r in cases]
            state={name:{metric:cluster_summary([r['full_truth'][name][metric] for r in entries],cfg,cfg['seed']+50)
                         for metric in ('nrmse','correlation','coverage90','coverage95')} for name in localization.TARGETS}
            increments={}
            for mask in cfg['teacher']['primary_masks']:
                increments[mask]={control:cluster_summary([r['masks'][mask]['controls'][control]['joint_increment'] for r in entries],cfg,cfg['seed']+51)
                                  for control in cfg['teacher']['controls']}
            masked={mask:{name:cluster_summary([r['masks'][mask]['truth_metrics'][name]['coverage95'] for r in entries],cfg,cfg['seed']+52)
                          for name in localization.TARGETS} for mask in cfg['teacher']['masks']}
            stability={key:cluster_summary([r['stability'][key] for r in entries],cfg,cfg['seed']+53) for key in ('driver_correlation_q05','driver_nrmse_q95')}
            result['laws'][law]=dict(state=state,shared_increments=increments,masked_coverage95=masked,stability=stability,
                                    physical_pass=all(r['physical_pass'] for r in entries),
                                    variance_decomposition={target:{kind:cluster_summary([r['variance_decomposition'][target][kind] for r in entries],cfg,cfg['seed']+54)
                                                                   for kind in ('state','parameter','observation')} for target in ('EEG','HbO','HbR')})
        primary=result['laws']['matched_model_calibration'];stress=result['laws']['misspecification_stress_test']
        result['checks']['physical']=primary['physical_pass'] and stress['physical_pass']
        result['checks']['matched_state_coverage']=all(cfg['teacher']['coverage95_min']<=primary['state'][t]['coverage95']['mean']<=cfg['teacher']['coverage95_max'] for t in localization.TARGETS)
        def accuracy(law):
            state=law['state']
            return bool(state['r']['nrmse']['mean']<=cfg['teacher']['maximum_driver_nrmse'] and state['r']['correlation']['mean']>=cfg['teacher']['minimum_driver_correlation'] and all(state[t]['nrmse']['mean']<=cfg['teacher']['maximum_clean_nrmse'] for t in localization.TARGETS[1:]))
        result['checks']['matched_state_accuracy']=accuracy(primary)
        result['checks']['stress_state_accuracy']=accuracy(stress)
        result['checks']['shared_information']=all(s['ci95'][0]>0 for targets in primary['shared_increments'].values() for s in targets.values())
        result['checks']['cross_parameter_stability']=bool(primary['stability']['driver_correlation_q05']['mean']>=cfg['teacher']['minimum_cross_parameter_driver_correlation'] and primary['stability']['driver_nrmse_q95']['mean']<=cfg['teacher']['maximum_cross_parameter_driver_nrmse'])
        if candidate=='U0_FIXED':
            entries=[r['laws']['matched_model_calibration'][candidate] for r in successes]
            sensitivity=dict(minimum_driver_correlation=min(v['driver_correlation'] for r in entries for v in r['sensitivity']),
                             maximum_driver_nrmse=max(v['driver_nrmse'] for r in entries for v in r['sensitivity']),
                             mean_worst_clean_nrmse_increase=float(np.mean([max(v['maximum_clean_nrmse_increase'] for v in r['sensitivity']) for r in entries])))
            result['sensitivity']=sensitivity
            result['checks']['fixed_parameter_sensitivity']=bool(sensitivity['minimum_driver_correlation']>=cfg['teacher']['minimum_cross_parameter_driver_correlation'] and sensitivity['maximum_driver_nrmse']<=cfg['teacher']['maximum_cross_parameter_driver_nrmse'] and sensitivity['mean_worst_clean_nrmse_increase']<=cfg['teacher']['maximum_sensitivity_clean_nrmse_increase'])
        else:
            entries=[r['laws'][law][candidate] for r in cases for law in ('matched_model_calibration','misspecification_stress_test')]
            result['maximum_mixture_mean_difference_in_noise_units']=max(r['mixture_mean_difference_in_noise_units'] for r in entries)
            result['maximum_mixture_relative_variance_difference']=max(r['mixture_relative_variance_difference'] for r in entries)
            result['checks']['parameter_mixture_resolution']=bool(result['maximum_mixture_mean_difference_in_noise_units']<=cfg['teacher']['mixture_mean_tolerance_in_noise_units'] and result['maximum_mixture_relative_variance_difference']<=cfg['teacher']['mixture_relative_variance_tolerance'] and max(r['training_grid_cdf_difference'] for r in entries)<=cfg['calibration']['grid_max_cdf_difference'])
        result['teacher_qualified']=all(result['checks'].values())
        result['parameter_physiology_claim']='NOT_GRANTED_BY_STATE_QUALIFICATION'
        summary['candidates'][candidate]=result
    summary['qualified_candidates']=[c for c,v in summary['candidates'].items() if v['teacher_qualified']]
    summary['measured_stage_eligible']=bool(summary['qualified_candidates'])
    return summary


def teacher_report(cfg,summary):
    lines=['# Step5A1 阶段报告：最小状态 teacher 资格','',
           '训练参数分布与留出 trial 的随机过程相互独立；正确配对、自身历史、任务模板和配对/时间位移 null 使用相同冻结训练参数分布。',
           'matched 用于同模型状态校准；stress 使用 Step 4 脉冲与确定性血流生成合同，其 coverage 为失配诊断，不称 SBC。',
           'U0 使用固定参数并接受 G/W/Z 的预设先验分位数敏感性检查；U1 使用训练后参数分布传播。U3 仅诊断，不进入候选选择。','',
           f"注册案例 {summary['independent_cases']}，成功 {summary['successful_cases']}，失败 {summary['failed_cases']}。失败种子与异常完整保留；以下数值仅描述成功案例，不能以成功子集授予资格。",'',
           '| 候选 | 合格 | 未通过检查 |','|---|---|---|']
    for c,v in summary['candidates'].items():lines.append(f"| {c} | {v['teacher_qualified']} | {', '.join(k for k,x in v['checks'].items() if not x) or '无'} |")
    lines+=['','| 候选 | matched r NRMSE | r corr | r 95%覆盖 | EEG | HbO | HbR |','|---|---:|---:|---:|---:|---:|---:|']
    for c,v in summary['candidates'].items():
        if not v['laws']:continue
        s=v['laws']['matched_model_calibration']['state']
        lines.append(f"| {c} | {s['r']['nrmse']['mean']:.4f} | {s['r']['correlation']['mean']:.4f} | "+' | '.join(f"{s[t]['coverage95']['mean']:.4f}" for t in localization.TARGETS)+' |')
    lines+=['','| 候选 | 遮挡目标 | 对照 | 配对 log-score 增量 | replicate bootstrap 95% CI |','|---|---|---|---:|---|']
    for c,v in summary['candidates'].items():
        if not v['laws']:continue
        for mask,controls in v['laws']['matched_model_calibration']['shared_increments'].items():
            for control,s in controls.items():lines.append(f"| {c} | {mask} | {control} | {s['mean']:.6f} | [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}] |")
    lines+=['',f"可进入后续最小 measured 检验的候选：**{', '.join(summary['qualified_candidates']) or '无'}**。",
            '所有区间先按独立 replicate 汇总；均值、方差分解、whole-modality 诊断、stress 指标和 U3 网格/相关性/边界质量详见 summary.json 与 case/u3 原始结果。',
            '参数恢复、状态恢复及不确定性风险排序是不同结论。上述状态资格不授予个体生理参数解释，也不自动验证精度加权。',
            '若无候选合格，Step5B 与全面 UQ 按冻结顺序记为未执行，不能用本阶段完成状态代替 scientific pass。','']
    return '\n'.join(lines)


def teacher_job(cfg,axis,replicate,calibration_dir):
    """Keep every registered failure instead of aborting/discarding its identity."""
    try:
        row,arrays=teacher_case(cfg,axis,replicate,calibration_dir)
        row['execution']='completed'
        return row,arrays
    except Exception as exc:
        return dict(axis=axis,replicate=replicate,execution='failed',
                    error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
                    seed_rule='case_seed(config, axis, replicate, stream); no redraw',
                    seed=cfg['seed']),None


def run_candidate_scope_review(cfg,run_dir,teacher_dir):
    """Correct an aggregation-scope error without rerunning or replacing cases."""
    teacher_dir=Path(teacher_dir).resolve()
    previous_manifest=json.loads((teacher_dir/'manifest.json').read_text())
    previous=json.loads((teacher_dir/'summary.json').read_text())
    if previous_manifest['stage']!='a1' or previous_manifest['execution']!='completed':
        raise ValueError('candidate scope review requires a completed A1 experiment')
    if cfg!=yaml.safe_load((teacher_dir/'resolved_config.yaml').read_text()):
        raise ValueError('review cannot change frozen configuration')
    rows=[json.loads(p.read_text()) for p in sorted(teacher_dir.glob('case_*.json'))]
    expected={(a,r) for a in ('G','W','Z','R') for r in range(cfg['teacher']['matching_replicates'])}
    if len(rows)!=len(expected) or {(r['axis'],r['replicate']) for r in rows}!=expected:
        raise ValueError('review requires every registered case identity')
    summary=teacher_summary(cfg,rows,previous['u3_diagnostic'])
    summary['analysis_correction']='Completeness belongs to each candidate panel. U1_W requires its own 60 W cases; U0 cross-scenario sensitivity requires all scenario panels. No numerical threshold, case, seed or result changed.'
    summary['source_teacher_run']=str(teacher_dir.relative_to(ROOT))
    for candidate,result in summary['candidates'].items():
        old=previous['candidates'][candidate]
        if old['laws']!=localization.jsonable(result['laws']) or any(old['checks'][key]!=value for key,value in result['checks'].items() if key!='complete_registered_cases'):
            raise RuntimeError('scope review unexpectedly changed scientific metrics')
    run_dir,manifest=start_stage(cfg,'a1_review',run_dir)
    write_json(run_dir/'summary.json',summary)
    (run_dir/'summary.md').write_text(teacher_report(cfg,summary)+'\n完整性按候选自身所需的注册面板核对；U1_W 的 60 个 W 案例全部完成。G/Z 的失败保留，不将它们错误地传播为 W 的失败。原聚合快照保留，所有指标、阈值和随机种子不变。\n')
    manifest.update(execution='completed',source_teacher_run=str(teacher_dir.relative_to(ROOT)),
                    source_summary_sha256=hashlib.sha256((teacher_dir/'summary.json').read_bytes()).hexdigest(),
                    calibration_run=previous_manifest['calibration_run'],new_inference_cases=0)
    write_json(run_dir/'manifest.json',manifest)
    return summary


def run_teacher(cfg,run_dir,calibration_dir,workers):
    validate_config(cfg);calibration_dir=Path(calibration_dir).resolve()
    previous=json.loads((calibration_dir/'summary.json').read_text())
    if previous['stage']!='a0' or not previous['numerical_inference_screen_pass']:
        raise ValueError('Step5A0 numerical inference prerequisite is not met')
    cal_cfg=yaml.safe_load((calibration_dir/'resolved_config.yaml').read_text())
    if any(cal_cfg[k]!=cfg[k] for k in ('model','axes','inference')):
        raise ValueError('teacher model/prior/inference differs from frozen calibration')
    run_dir,manifest=start_stage(cfg,'a1',run_dir);started=time.time();rows=[]
    manifest['calibration_run']=str(calibration_dir.relative_to(ROOT))
    manifest['calibration_summary_sha256']=hashlib.sha256((calibration_dir/'summary.json').read_bytes()).hexdigest()
    manifest.update(workers=workers,process_start_method='spawn')
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            jobs={pool.submit(teacher_job,cfg,a,r,calibration_dir):(a,r) for a in ('G','W','Z','R') for r in range(cfg['teacher']['matching_replicates'])}
            for job in as_completed(jobs):
                row,arrays=job.result();rows.append(row);stem=f"case_{row['axis']}_{row['replicate']:02d}"
                write_json(run_dir/f'{stem}.json',row)
                if arrays is not None:np.savez_compressed(run_dir/f'{stem}.npz',**arrays)
                print(f"a1 {len(rows)}/{len(jobs)} {stem} {row['execution']}",flush=True)
        rows.sort(key=lambda r:(r['axis'],r['replicate']))
        u3=u3_diagnostic(cfg,calibration_dir,run_dir,workers)
        summary=teacher_summary(cfg,rows,u3);write_json(run_dir/'summary.json',summary)
        (run_dir/'summary.md').write_text(teacher_report(cfg,summary))
        manifest.update(execution='completed',completed_cases=len(rows),elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(execution='failed',completed_cases=len(rows),error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.time()-started)
        raise
    finally:write_json(run_dir/'manifest.json',manifest)
    return summary


def preprocess_native_trial(eeg, intensity_pairs, *, eeg_hz=200., fnirs_hz=10.,
                            target_hz=4., mask_name=None, gap_seconds=4.):
    """Trial-local Step5B features with the target hidden before every transform.

    Inputs are native EEG and positive intensity [time, pair, wavelength].
    Interpolation uses visible samples only. The scoring branch calls this
    function separately without a mask. Learned normalization, channel choice,
    PCA and noise scales must subsequently be fitted on training trials only.
    No canonical full-record standardized/filtered cache is accepted here.
    """
    from fractions import Fraction
    from scipy.signal import butter, resample_poly, sosfiltfilt
    from experiments.evaluate_shared_neural_driver_unified import _downsample_eeg_power
    from src.data.homer2_preprocessing import apply_homer2_aligned_contract

    eeg=np.asarray(eeg,dtype=float); intensity_pairs=np.asarray(intensity_pairs,dtype=float)
    if eeg.ndim!=2 or intensity_pairs.ndim!=3 or intensity_pairs.shape[2]!=2:
        raise ValueError('native trial requires EEG [time,channel] and intensity [time,pair,2]')
    duration=len(eeg)/eeg_hz
    if not np.isclose(duration,len(intensity_pairs)/fnirs_hz):
        raise ValueError('native trial clocks have different durations')
    if mask_name not in (None,'center_EEG','center_fNIRS','whole_EEG','whole_fNIRS'):
        raise ValueError('unsupported trial mask')
    if not 0<gap_seconds<duration:
        raise ValueError('mask gap must fit within the trial')
    count=int(round(duration*target_hz)); output_mask=np.ones((count,3),dtype=bool)
    if mask_name:
        _,hidden,columns=masked_input(np.zeros((count,3)),mask_name,int(round(gap_seconds*target_hz)))
        output_mask[np.ix_(hidden,columns)]=False

    def restrict(values,hz,modality):
        values=np.array(values,copy=True); hidden=np.zeros(len(values),dtype=bool)
        if mask_name and mask_name.endswith(modality):
            if mask_name.startswith('whole'):return None
            left=(count-int(round(gap_seconds*target_hz)))//2/target_hz
            right=left+int(round(gap_seconds*target_hz))/target_hz
            hidden=(np.arange(len(values))/hz>=left)&(np.arange(len(values))/hz<right)
        values[hidden]=np.nan
        flat=values.reshape(len(values),-1);clock=np.arange(len(values))
        for j in range(flat.shape[1]):
            visible=np.isfinite(flat[:,j])
            if visible.sum()<2:raise ValueError('insufficient visible native samples')
            flat[:,j]=np.interp(clock,clock[visible],flat[visible,j])
        return values

    input_eeg=restrict(eeg,eeg_hz,'EEG');input_fnirs=restrict(intensity_pairs,fnirs_hz,'fNIRS')
    if input_eeg is None:
        eeg_features=np.full((count,eeg.shape[1]),np.nan)
    else:
        if not np.isclose(eeg_hz/target_hz,round(eeg_hz/target_hz)):
            raise ValueError('EEG power requires an integer sampling ratio')
        filtered=sosfiltfilt(butter(4,[1.,45.],btype='bandpass',fs=eeg_hz,output='sos'),input_eeg,axis=0)
        power=_downsample_eeg_power(filtered,eeg_hz,target_hz)
        eeg_features=np.log(np.maximum(power,1e-12))
        eeg_features[~output_mask[:,0]]=np.nan
    if input_fnirs is None:
        fnirs_features=np.full((count,intensity_pairs.shape[1],2),np.nan)
    else:
        transformed=apply_homer2_aligned_contract(input_fnirs,dataset_id='eeg_fnirs_single_trial',
                         sample_rate_hz=fnirs_hz,entry_stage='raw_intensity',wavelengths_nm=(760.,850.))
        ratio=Fraction(target_hz/fnirs_hz).limit_denominator(1000)
        fnirs_features=resample_poly(transformed.values,ratio.numerator,ratio.denominator,axis=0)
        fnirs_features=fnirs_features.reshape(count,intensity_pairs.shape[1],2)
        fnirs_features[~output_mask[:,1]]=np.nan
    return dict(eeg_log_power=eeg_features,fnirs=fnirs_features,observation_mask=output_mask)


def report_step5(run_dir):
    """Summarize completed B evidence and record the conditional UQ decision."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dir=Path(run_dir).resolve()
    manifest=json.loads((run_dir/'manifest.json').read_text())
    cfg=yaml.safe_load((run_dir/'resolved_config.yaml').read_text());validate_config(cfg)
    summary=json.loads((run_dir/'summary.json').read_text())
    if manifest['stage']!='b' or manifest['execution']!='completed':
        raise ValueError('final stage review requires completed measured B evidence')
    if summary['uq_stage_eligible']:
        raise ValueError('qualified measured teacher requires actual UQ; cannot report a blocked stage')
    teacher_dir=ROOT/manifest['teacher_run']
    teacher_manifest=json.loads((teacher_dir/'manifest.json').read_text())
    teacher=json.loads((teacher_dir/'summary.json').read_text())
    teacher_source=ROOT/teacher_manifest.get('source_teacher_run',manifest['teacher_run'])
    calibration_dir=ROOT/manifest['calibration_run']
    calibration=json.loads((calibration_dir/'summary.json').read_text())
    info={}
    for axis in cfg['axes']:
        posterior=[json.loads(p.read_text())['joint_posterior'] for p in sorted(calibration_dir.glob(f'case_{axis}_*.json'))]
        widths=np.array([p['upper95']-p['lower95'] for p in posterior]);material=cfg['axes'][axis]['material_change']
        info[axis]=dict(material_change=material,mean_interval_width=float(widths.mean()),
            fraction_width_within_material_change=float(np.mean(widths<=material)),
            mean_boundary_mass=float(np.mean([p['boundary_mass'] for p in posterior])),
            mean_mle_map_separation_fraction=float(np.mean([p['mode_separation_fraction'] for p in posterior])),
            individual_physiology_claim=False)
    write_json(run_dir/'parameter_information_diagnostic.json',info)
    posteriors={p.stem.replace('posterior_',''):json.loads(p.read_text()) for p in sorted(run_dir.glob('posterior_subject_*.json'))}
    numerical={s:dict(points=len(p['grid']),cdf_difference=p['grid_cdf_difference'],
        quadrature_cdf_difference=p['quadrature_cdf_difference'],posterior=p['posterior']) for s,p in posteriors.items()}
    write_json(run_dir/'measured_posterior_diagnostic.json',numerical)
    decision=dict(stage='uq',execution='not_executed',scientific_verdict='not_evaluated',
        reason='NO_MEASURED_CORE_TEACHER',prerequisite_run=str(run_dir.relative_to(ROOT)),
        qualified_candidates=summary['qualified_candidates'],new_measured_data_reads=False,
        teacher_training_exports_created=0,precision_weighting_tested=False,
        failed_primary_checks=[k for k,v in summary['candidates']['U1_W']['checks'].items() if not v])
    write_json(run_dir/'uq_stage_decision.json',decision)
    uq_lines=['# 第四阶段：全面 UQ 未执行','',
        '**Step5B 未获得合格实测核心 teacher，因此按 ssm_next.md 的阶段顺序停止。全面 UQ 的统计结论未被检验。**','',
        '前置失败：'+', '.join(decision['failed_primary_checks'])+'。详见 [Step5B 报告](summary.md)。',
        '没有用跨被试/跨模态森林图、ICC、conformal 或 precision weighting 绕过核心资格；这些分析未运行。','',
        'A1/B 已保存的状态方差、参数均值间方差、观测噪声与遮挡覆盖属于资格诊断。它们不是合格 teacher 的全面 UQ，也不证明真实潜在轨迹覆盖。',
        'clean teacher 方差为 E[Var(h(x)|D,phi)] + Var(E[h(x)|D,phi])；仅带噪观测预测再加 Student-t 观测方差。状态后验方差条件于固定过程模型，不包含全部模型失配。','',
        '若未来核心资格成立，校准对象应继续明确为已知被试和 session 中的新 trial。普通 cross-fit 不被宣称具有标准 split-conformal 的有限样本保证；风险排序用于判断是否可测试精度加权，不代替 teacher 资格。','',
        '本轮未创建可供训练的合格 teacher 导出，未训练 tokenizer。监督目标仍是同模态观测空间 clean teacher 切片，r 仅作共享过程诊断；默认统一权重。','',
        'subjects 19–23 未启动 replication；subjects 24–29 未读取。停止来自实测资格结果，不是等待额外授权。','']
    (run_dir/'uq_stage_report.md').write_text('\n'.join(uq_lines))
    lines=['# Step5 分阶段详细复核','',
        '**A0、A1 和 B 已执行；B 未取得合格实测 teacher，第四阶段全面 UQ 按前置条件未执行。四个实验阶段没有全部运行。**','',
        '## Step5A0：推断一致性与参数信息','',
        f"{calibration['independent_cases']} 个独立同模型案例通过冻结数值和参数校准 screen。联合 Student-t 观测积分仍使用高斯状态闭合近似；通过有限面板不证明全局精确后验。",'',
        '| 参数轴 | Oracle 95%覆盖 | 旧评分 95%覆盖 | 联合似然 95%覆盖 | rank KS p | 平均95%区间宽 / 实质变化尺度 |',
        '|---|---:|---:|---:|---:|---|']
    for axis,v in calibration['axes'].items():
        lines.append(f"| {axis} | {v['oracle']['coverage95']:.4f} | {v['legacy_generalized_posterior']['coverage95']:.4f} | {v['joint_posterior']['coverage95']:.4f} | {v['joint_posterior']['rank_ks_p']:.5f} | {info[axis]['mean_interval_width']:.4f} / {info[axis]['material_change']:.4f} |")
    lines+=['','这些平均区间宽于预设实质变化尺度，参数校准不等于个体参数被精确辨识。参数先验定义在 G/W/Z 的注册坐标；没有通过 logistic 坐标改变先验。',
        '短序列 PF 参考的似然精度通过，但 W 的祖先多样性未通过，因此不使用其粒子平滑路径作状态真值。详见 A0 原报告。','',
        '## Step5A1：最小状态 teacher','',
        f"240 个注册案例成功 {teacher['successful_cases']}、失败 {teacher['failed_cases']}。U1 按自身参数轴的 60 个案例判断完整性；U0 还需完整跨情景敏感性。合格候选为 {', '.join(teacher['qualified_candidates'])}。",'',
        '| 候选 | 主面板成功案例 | r NRMSE | r corr | r/EEG/HbO/HbR 95%覆盖 | 未通过检查 |',
        '|---|---:|---:|---:|---|---|']
    for c,v in teacher['candidates'].items():
        state=v['laws']['matched_model_calibration']['state']
        cover='/'.join(f"{state[t]['coverage95']['mean']:.4f}" for t in localization.TARGETS)
        lines.append(f"| {c} | {v['independent_cases']} | {state['r']['nrmse']['mean']:.4f} | {state['r']['correlation']['mean']:.4f} | {cover} | {', '.join(k for k,ok in v['checks'].items() if not ok) or '无'} |")
    lines+=['','U1_W 的双向四类 null 增量区间下界全部为正。U0 的固定参数敏感性未通过；G/Z 的失败案例保留，不补抽、不把成功子集当完整面板。',
        '同种子的 2400 条生成流审计发现 4 条转移失败；2/8/32 子步仍失败，不能只归因于粗积分。另有 null 输入氧提取数值域异常。A0 的有限面板不能排除这些支持域问题。',
        'U3 的三个诊断案例均未通过 9³/13³ 网格 CDF 分辨率，不参与候选选择。stress coverage 仅描述 Step4 失配条件，不称 SBC。',
        '初次 A1 聚合错误地将 G/Z 不完整传播到 W；独立 review 修正候选作用域，指标、阈值和随机种子均未改变，旧快照保留。','',
        '## Step5B：最小实测状态检验','',
        f"subjects 01–18，三个已有 session，每 session 8 train / 2 heldout；主候选完成 {summary['successful_heldout_trials']}/108，完整六条留出结果的被试 {len(summary['complete_subjects'])}/18。",'',
        '主候选仅 U1_W；U0 保留基线身份。中心 4 秒遮挡、整模态缺失、同模态上下文、训练任务模板与两类配对 null 均保留。',
        '输入在滤波、功率、OD/MBLL 和重采样前遮挡；训练投影、通道和噪声规则固定后才处理留出 trial。单个原始 fNIRS pair 不合法不再排除其他合法 pair，通道资格只用训练样本判断，留出不换通道。',
        'v1 的均匀网格未解析边界质量；v2 仅在同一完整支持域插入网格点，旧端点与尾部节点保留。首次 v2 写出 subject_01 诊断时遇到 -inf JSON 异常；恢复运行仅修复诊断序列化并补齐缺失结果，来源哈希与旧失败均保留。','',
        '| subject | 网格点 | CDF加密差 | order13/17差 | W均值 | 边界带质量 |',
        '|---|---:|---:|---:|---:|---:|']
    for s,v in numerical.items():
        p=v['posterior'];q=v['quadrature_cdf_difference']
        lines.append(f"| {s} | {v['points']} | {v['cdf_difference']:.6f} | {q if q is not None else '未预定'} | {p['mean']:.6f} | {p['boundary_mass']:.6f} |")
    lines+=['','完整实测 null、误差、观测覆盖、失败条件见 [B 阶段报告](summary.md)；机器可读结果见 [summary.json](summary.json)。',
        '即使参数网格收敛，边界后验也不授予个体生理解释。数值解析、实测共享增量和状态资格分别判断。','',
        '## 第四阶段及后续边界','',
        '没有合格实测 teacher，全面 UQ 未执行，详见 [第四阶段报告](uq_stage_report.md)。不以人为扩边、增加生理自由参数或改写 tokenizer 目标绕过本次负结果。',
        '后续应先在同一状态结构中定位实测观测映射、时间/频带和过程/观测噪声失配，再冻结新的最小诊断；这不是本轮已取得的科学结论。',
        'Step1–4 代码、冻结合同和负证据保留；本轮没有 replication、protected evaluation、tokenizer 训练或外部发布。','']
    (run_dir/'stage_review.md').write_text('\n'.join(lines))

    controls=cfg['teacher']['controls'];x=np.arange(len(controls))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,(source,label) in enumerate(((teacher,'A1 matched: 60 independent W cases'),(summary,'B: complete-subject subset'))):
        v=source['candidates']['U1_W']
        scores=v['laws']['matched_model_calibration']['shared_increments'] if row==0 else {m:v['masks'][m]['controls'] for m in cfg['teacher']['primary_masks']}
        for col,mask in enumerate(cfg['teacher']['primary_masks']):
            mean=np.array([scores[mask][c]['mean'] for c in controls]);ci=np.array([scores[mask][c]['ci95'] for c in controls])
            ax=axes[row,col];ax.errorbar(x,mean,yerr=np.maximum([mean-ci[:,0],ci[:,1]-mean],0),fmt='o',capsize=4,color='#276FBF' if row==0 else '#C35242')
            ax.axhline(0,color='#555555',ls='--',lw=1)
            ax.set(xticks=x,xticklabels=['History','History + task','Pairing null','Shift null'],ylabel='Joint minus control log score',title=f'{label}\n{mask}')
            ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('U1_W: synthetic qualification does not establish measured shared information',fontweight='bold')
    for suffix in ('png','pdf'):fig.savefig(run_dir/f'shared_information_diagnostic.{suffix}',dpi=180)
    plt.close(fig)
    inputs=[run_dir/'summary.json',run_dir/'manifest.json',teacher_dir/'summary.json',calibration_dir/'summary.json']
    provenance=dict(stage='post_run_stage_review',execution='completed',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        measured_arrays_read=False,uq=decision)
    write_json(run_dir/'stage_review_manifest.json',provenance)
    return provenance


def load_measured_config(path=MEASURED_CONFIG):
    spec=yaml.safe_load(Path(path).read_text())
    if spec['schema'] not in ('step5b_measured_v1','step5b_measured_v2'):raise ValueError('wrong measured schema')
    loaded=[]
    for name in ('base_config','metadata_config'):
        source=ROOT/spec[name]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=spec[name+'_sha256']:
            raise ValueError(f'{name} digest mismatch')
        loaded.append(yaml.safe_load(source.read_text()))
    base,metadata=loaded;validate_config(base)
    if (spec['primary']!='U1_W' or spec['candidates']!=['U0_FIXED','U1_W'] or
        spec['window_seconds']!=30. or spec['sampling_hz']!=4. or spec['baseline_seconds']!=5. or
        spec['window_offset_seconds']!=-5. or spec['posterior']['axis']!='W'):
        raise ValueError('measured candidate/timing contract drift')
    if (metadata['data']['subjects']!=base['measured']['subjects'] or
        [s['record_id'] for s in metadata['data']['sessions']]!=base['measured']['sessions'] or
        metadata['data']['target_label']!=base['measured']['condition'] or
        spec['preprocessing']['fnirs_array_key']!='native_input_fnirs'):
        raise ValueError('measured data scope or native-array contract drift')
    if spec['qualification']['complete_subjects']!=18 or spec['qualification']['complete_heldout_trials_per_subject']!=6:
        raise ValueError('registered measured counts changed')
    if spec['schema']=='step5b_measured_v2' and spec['preprocessing'].get('fnirs_pair_validity')!='positive_finite_in_all_training_samples_then_check_selected_pair_on_heldout':
        raise ValueError('v2 requires training-only pair eligibility and selected heldout support')
    return base,spec,metadata


def require_measured_teacher(teacher_dir,base,spec):
    teacher_dir=Path(teacher_dir).resolve()
    manifest=json.loads((teacher_dir/'manifest.json').read_text())
    summary=json.loads((teacher_dir/'summary.json').read_text())
    if manifest['execution']!='completed' or manifest['stage'] not in ('a1','a1_review'):
        raise ValueError('measured stage needs completed synthetic teacher evidence')
    if not summary['candidates'][spec['primary']]['teacher_qualified']:
        raise ValueError('primary teacher has not passed its complete synthetic panel')
    if yaml.safe_load((teacher_dir/'resolved_config.yaml').read_text())!=base:
        raise ValueError('teacher contract differs from measured base model')
    for name in ('src/inference/t3a_balloon_joint_ssm.py','src/inference/t3a_balloon_robust_ssm.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=manifest['source_sha256'][name]:
            raise ValueError('inference changed since synthetic qualification')
    return manifest,summary


def robust_mad(values,axis=0):
    values=np.asarray(values,dtype=float)
    median=np.median(values,axis=axis,keepdims=True)
    return 1.482602218505602*np.median(abs(values-median),axis=axis)


def student_difference_mad(nu):
    """Median absolute difference of two independent unit-scale Student draws."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    def mass(q):
        return quad(lambda x:student_t.pdf(x,nu)*(student_t.cdf(x+q,nu)-student_t.cdf(x-q,nu)),
                    -np.inf,np.inf,epsabs=1e-10)[0]-.5
    return float(brentq(mass,.01,10.))


def first_difference_noise(trials,constant):
    differences=np.concatenate([np.diff(y,axis=0) for y in trials],axis=0)
    return robust_mad(differences)/1.482602218505602/constant


def reference_observation_gauge(base,spec):
    p,c=localization.model(base,'G',0.)
    _,a=core.rk4_transition_with_jacobian(np.zeros(6),p,c)
    covariance=np.diag(np.square(c.initial_state_std));q=np.diag(np.square(p.fixed.process_std))*c.dt
    observation=core.BalloonObservationSpec().resolved(p.fixed);h=core._observation_physical_matrix(p,observation)
    variances=[]
    for i in range(round(spec['window_seconds']*spec['sampling_hz'])):
        if i:covariance=a@covariance@a.T+q
        _,physical=core.transformed_gaussian_moments(np.zeros(6),covariance)
        variances.append(np.diag(h@physical@h.T))
    variance=np.mean(variances,axis=0)
    return dict(eeg=float(np.sqrt(variance[0])),fnirs_common=float(np.sqrt(np.mean(variance[1:]))),
                coordinate_sd=np.sqrt(variance),meaning='fixed reference covariance gauge, not fitted physiological parameters')


def noise_estimator_preflight(base,spec,calibration_dir):
    constant=student_difference_mad(base['model']['student_nu']);estimates=[]
    for path in sorted(Path(calibration_dir).glob('case_*.npz')):
        with np.load(path) as arrays:noise=arrays['observations']-arrays['truth_clean']
        estimates.append(first_difference_noise([noise],constant))
    if len(estimates)!=180:raise ValueError('noise check needs the complete retained calibration panel')
    relative=np.array(estimates)/np.array(base['model']['observation_scale'])-1
    mean_bias=relative.mean(axis=0);median_error=np.median(abs(relative),axis=0)
    passed=bool(np.max(abs(mean_bias))<=spec['noise']['maximum_absolute_mean_relative_bias'] and
                np.max(median_error)<=spec['noise']['maximum_median_absolute_relative_error'])
    return dict(known_noise_cases=len(estimates),difference_mad_constant=constant,mean_relative_bias=mean_bias,
                median_absolute_relative_error=median_error,passed=passed,
                interpretation='Checks known iid observation innovations only; filtered measured first differences remain an initial scale estimate.')


def fit_measured_projection(features,base,spec,pair_eligible=None):
    baseline=round(spec['baseline_seconds']*spec['sampling_hz'])
    eeg=np.concatenate([v['eeg_log_power'] for v in features])
    center=np.median(eeg,axis=0);scale=np.maximum(robust_mad(eeg),1e-8)
    normalized=(eeg-center)/scale;pca_center=normalized.mean(axis=0)
    _,singular,vectors=np.linalg.svd(normalized-pca_center,full_matrices=False)
    loading=vectors[0]
    if loading.sum()<0:loading=-loading
    projected=[((v['eeg_log_power']-center)/scale-pca_center)@loading for v in features]
    projected=[v-v[:baseline].mean() for v in projected]
    pc_scale=max(float(robust_mad(np.concatenate(projected))),1e-8)
    fnirs=[v['fnirs']-v['fnirs'][:baseline].mean(axis=0) for v in features]
    signal=np.maximum(robust_mad(np.concatenate(fnirs)),1e-12)
    difference=np.maximum(robust_mad(np.concatenate([np.diff(v,axis=0) for v in fnirs])),1e-12)
    pair_scores=(signal/difference).mean(axis=1)
    selection_scores=pair_scores
    if pair_eligible is not None:
        pair_eligible=np.asarray(pair_eligible,dtype=bool)
        if pair_eligible.shape!=pair_scores.shape or not pair_eligible.any():raise ValueError('no valid training fNIRS pair')
        selection_scores=np.where(pair_eligible,pair_scores,-np.inf)
    pair=int(np.argmax(selection_scores))
    common=max(float(robust_mad(np.concatenate([v[:,pair].reshape(-1) for v in fnirs]))),1e-12)
    gauge=reference_observation_gauge(base,spec)
    return dict(eeg_center=center,eeg_feature_scale=scale,pca_center=pca_center,loading=loading,
                eeg_factor=gauge['eeg']/pc_scale,fnirs_pair=pair,fnirs_factor=gauge['fnirs_common']/common,
                pair_scores=pair_scores,pair_eligible=pair_eligible,
                pca_explained_fraction=float(singular[0]**2/np.sum(singular**2)),
                gauge=gauge,baseline_samples=baseline,training_trials=len(features),task_labels_used=False)


def apply_measured_projection(features,projection):
    eeg=features['eeg_log_power'];fnirs=features['fnirs'][:,projection['fnirs_pair']]
    baseline=projection['baseline_samples']
    if np.isfinite(eeg).any():
        driver=((eeg-projection['eeg_center'])/projection['eeg_feature_scale']-projection['pca_center'])@projection['loading']
        driver=(driver-driver[:baseline].mean())*projection['eeg_factor']
    else:driver=np.full(len(eeg),np.nan)
    if np.isfinite(fnirs).any():fnirs=(fnirs-fnirs[:baseline].mean(axis=0))*projection['fnirs_factor']
    return np.column_stack((driver,fnirs))


def prepare_measured_subject(base,spec,metadata,subject,noise_constant):
    """The only Step5 native-array entry: enforce subject/record scope first."""
    if subject not in base['measured']['subjects'] or subject not in metadata['data']['subjects']:
        raise ValueError('subject outside the Step5 development boundary')
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_eeg_record
    from experiments.build_clean_eeg_fnirs_cache import _pair_single_trial_wavelengths
    from experiments.evaluate_t3_multisession_loso import _sha256
    index=CleanPhysiologyCacheIndex(ROOT/metadata['data']['cache_root'])
    records=sorted([r for r in index.records if r.dataset_id==base['measured']['dataset_id']
                    and r.canonical_subject_id==subject and r.base_record_id in base['measured']['sessions']],key=lambda r:r.base_record_id)
    if len(records)!=3 or [r.base_record_id for r in records]!=base['measured']['sessions']:
        raise ValueError('subject does not have the exact three admitted records')
    raw=[];identities=[];source_files={};eeg_names=None;pair_names=None
    for record in records:
        native=load_native_eeg_record(ROOT,record)
        arrays=index.load_record_arrays(record)
        values=arrays[spec['preprocessing']['fnirs_array_key']]
        paired,names=_pair_single_trial_wavelengths(values,arrays['native_channel_names'])
        if native.sample_rate_hz!=spec['native_eeg_hz'] or record.sample_rate_hz!=spec['native_fnirs_hz']:
            raise ValueError('native sampling rate changed')
        if eeg_names is not None and (eeg_names!=native.channel_names or pair_names!=names):
            raise ValueError('cross-session channel identity mismatch')
        eeg_names=native.channel_names;pair_names=names
        for path in (native.source_path,record.npz_path):
            key=str(path.relative_to(ROOT))
            if key not in source_files:source_files[key]=_sha256(path)
        events=sorted([e for e in index.events_by_join_key[record.join_key] if e['label']==base['measured']['condition']],key=lambda e:int(e['event_index']))
        if len(events)!=10:raise ValueError('exact ten MA trials required')
        for position,event in enumerate(events):
            starts={modality:int(round((float(event[f'{modality}_time_ms'])/1000+spec['window_offset_seconds'])*hz))
                    for modality,hz in [('eeg',native.sample_rate_hz),('fnirs',record.sample_rate_hz)]}
            e=native.values[starts['eeg']:starts['eeg']+round(spec['window_seconds']*native.sample_rate_hz)].copy()
            f=paired[starts['fnirs']:starts['fnirs']+round(spec['window_seconds']*record.sample_rate_hz)].copy()
            if e.shape[0]!=6000 or f.shape[0]!=300:raise ValueError('native trial support changed after metadata check')
            if not np.isfinite(e).all() or not np.isfinite(f).all() or (spec['schema']=='step5b_measured_v1' and np.any(f<=0)):
                raise ValueError(f"invalid native observation support: {record.join_key} event {event['event_index']}")
            raw.append((e,f));identities.append(dict(dataset_id=record.dataset_id,subject=subject,session=record.base_record_id,
                event_index=int(event['event_index']),ma_trial_position=position,condition=event['label'],
                heldout=position in base['measured']['heldout_trial_positions'],native_start_samples=starts,
                eeg_time_ms=float(event['eeg_time_ms']),fnirs_time_ms=float(event['fnirs_time_ms']),
                sample_id=f"{record.join_key}|event={event['event_index']}|offset=-5.0|duration=30.0"))
    train=[i for i,v in enumerate(identities) if not v['heldout']];held=[i for i,v in enumerate(identities) if v['heldout']]
    if len(train)!=24 or len(held)!=6:raise ValueError('subject train/heldout inventory is not 24/6')
    features={i:preprocess_native_trial(*raw[i],target_hz=spec['sampling_hz']) for i in train}
    eligible=np.logical_and.reduce([np.all(raw[i][1]>0,axis=(0,2)) for i in train])
    projection=fit_measured_projection([features[i] for i in train],base,spec,eligible if spec['schema']=='step5b_measured_v2' else None)
    training=np.array([apply_measured_projection(features[i],projection) for i in train])
    targets=[];masked=[];templates=[];donors=[]
    for position,i in enumerate(held):
        if np.any(raw[i][1][:,projection['fnirs_pair']]<=0):
            raise ValueError(f"selected heldout fNIRS pair has invalid support: {identities[i]['sample_id']}")
        targets.append(apply_measured_projection(preprocess_native_trial(*raw[i],target_hz=spec['sampling_hz']),projection))
        masked.append([apply_measured_projection(preprocess_native_trial(*raw[i],target_hz=spec['sampling_hz'],mask_name=mask),projection)
                       for mask in base['teacher']['masks']])
        local=[j for j,index_i in enumerate(train) if identities[index_i]['session']==identities[i]['session']]
        if len(local)!=8:raise ValueError('task control requires exactly eight same-session training trials')
        templates.append(training[local].mean(axis=0));donors.append(training[local[position%8]])
    estimate=first_difference_noise(training,noise_constant)
    effective=copy.deepcopy(base);effective['model']['steps']=120
    effective['model']['observation_scale']=np.maximum(estimate,base['model']['observation_scale']).tolist()
    detail=dict(subject=subject,execution='completed',projection=projection,source_files=source_files,trial_inventory=identities,
                selected_fnirs_pair=pair_names[projection['fnirs_pair']],eeg_channels=eeg_names,
                training_ineligible_fnirs_pairs=[pair_names[i] for i in np.flatnonzero(~eligible)],
                sampling_hz=spec['sampling_hz'],coordinate_names=['EEG_PCA1',pair_names[projection['fnirs_pair']]+'_HbO',pair_names[projection['fnirs_pair']]+'_HbR'],
                coordinate_units='fixed operational model gauge; paired fNIRS common scale, not absolute concentration',
                observation_noise_estimate=estimate,observation_scale=effective['model']['observation_scale'],
                data_branch='native EEG raw_with_ocular_artifact; native fNIRS intensity; trial-local Step5 transforms',
                heldout_fitting_calls=0,heldout_parameter_updates=0)
    payload=dict(training=training,targets=np.array(targets),masked_inputs=np.array(masked),templates=np.array(templates),donors=np.array(donors),
                 event_relative_time_s=np.arange(120)/spec['sampling_hz']+spec['window_offset_seconds'])
    return detail,payload,effective


def measured_likelihood_chunk(config,training,points,order):
    scores=[]
    for value in points:
        p,c=localization.model(config,'W',float(value))
        scores.append(sum(joint.parameter_log_likelihood(y,p,config=c,quadrature_order=order) for y in training))
    return scores


def fit_measured_posteriors(base,spec,prepared,workers,run_dir):
    """One adaptive W grid per subject; independent grid jobs share a pool."""
    fit={};failed={};pending=list(prepared);n=spec['posterior']['grid_points'];known={}
    while pending:
        grid=np.linspace(*base['axes']['W']['bounds'],n)
        scores={subject:np.full(n,np.nan) for subject in pending}
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            jobs={}
            for subject in pending:
                if subject in known:
                    scores[subject][::2]=known[subject];indices=np.arange(1,n,2)
                else:indices=np.arange(n)
                config=prepared[subject]['config'];training=prepared[subject]['arrays']['training']
                for offset in range(0,len(indices),4):
                    part=indices[offset:offset+4]
                    jobs[pool.submit(measured_likelihood_chunk,config,training,grid[part],base['inference']['quadrature_order'])]=(subject,part)
            for job in as_completed(jobs):
                subject,indices=jobs[job]
                try:scores[subject][indices]=job.result()
                except Exception as exc:
                    failed[subject]=dict(subject=subject,execution='failed',stage='training_posterior',
                                         error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        next_pending=[]
        for subject in pending:
            if subject in failed:continue
            values=scores[subject]
            if not np.isfinite(values).all():raise RuntimeError('training likelihood grid has missing results')
            post=localization.posterior_grid(base,'W',grid,values)
            coarse=localization.posterior_grid(base,'W',grid[::2],values[::2])
            difference=float(max(abs(post['cdf']-np.interp(grid,grid[::2],coarse['cdf']))))
            result=dict(subject=subject,grid=grid,parameter_log_likelihood=values,grid_cdf_difference=difference,
                        posterior=localization.posterior_summary(base,'W',grid,values,0.),quadrature_cdf_difference=None)
            # The unknown measured theta is not zero: strip simulation-only fields.
            for key in ('rank_u','covered95'):result['posterior'].pop(key,None)
            if difference>spec['posterior']['maximum_cdf_difference'] and n<spec['posterior']['maximum_grid_points']:
                known[subject]=values;next_pending.append(subject)
            else:fit[subject]=result
        print(f'b training grid {n}: resolved-or-capped {len(fit)}, failed {len(failed)}, refining {len(next_pending)}',flush=True)
        pending=next_pending;n=min(2*n-1,spec['posterior']['maximum_grid_points'])
    selected=[base['measured']['subjects'][i] for i in spec['posterior']['quadrature_check_subject_positions']]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        jobs={};checked={subject:np.full(len(fit[subject]['grid']),np.nan) for subject in selected if subject in fit}
        for subject in checked:
            config=prepared[subject]['config'];training=prepared[subject]['arrays']['training'];grid=fit[subject]['grid']
            for i in range(0,len(grid),4):
                jobs[pool.submit(measured_likelihood_chunk,config,training,grid[i:i+4],base['inference']['quadrature_check_order'])]=(subject,i)
        for job in as_completed(jobs):
            subject,i=jobs[job]
            try:
                values=job.result();checked[subject][i:i+len(values)]=values
            except Exception as exc:
                failed[subject]=dict(subject=subject,execution='failed',stage='quadrature_reference',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        for subject,values in checked.items():
            if subject in failed:fit.pop(subject,None);continue
            result=fit[subject];grid=result['grid']
            old=localization.posterior_grid(base,'W',grid,result['parameter_log_likelihood'])
            new=localization.posterior_grid(base,'W',grid,values)
            result['quadrature_cdf_difference']=float(max(abs(old['cdf']-new['cdf'])))
            result['refined_quadrature_log_likelihood']=values
    for subject,result in fit.items():write_json(Path(run_dir)/f'posterior_{subject}.json',result)
    for subject,result in failed.items():write_json(Path(run_dir)/f'failure_{subject}.json',result)
    return fit,failed


def refine_measured_posterior_mass(base,spec,prepared,fit,workers,run_dir):
    """Resolve a narrow boundary posterior without truncating any support."""
    refinement=spec['posterior']['local_refinement'];pending=list(fit)
    for subject in fit:
        fit[subject]['uniform_grid_cdf_difference']=fit[subject]['grid_cdf_difference']
        fit[subject]['local_refinement_history']=[]
    for iteration in range(refinement['maximum_rounds']):
        if not pending:break
        updates={};errors={}
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            jobs={}
            for subject in pending:
                record=fit[subject];grid=np.array(record['grid']);ll=np.array(record['parameter_log_likelihood'])
                density=ll+localization.prior(base,'W').logpdf(grid)
                active=np.maximum(density[:-1],density[1:])>=max(density)-refinement['log_density_drop']
                middle=(grid[:-1]+grid[1:])/2;middle=middle[active]
                if len(grid)+len(middle)>refinement['maximum_total_grid_points']:
                    record['local_refinement_stopped']='POINT_BUDGET';continue
                updates[subject]=dict(old_grid=grid,old_ll=ll,added_grid=middle,added_ll=np.full(len(middle),np.nan))
                for i in range(0,len(middle),4):
                    jobs[pool.submit(measured_likelihood_chunk,prepared[subject]['config'],prepared[subject]['arrays']['training'],middle[i:i+4],base['inference']['quadrature_order'])]=(subject,i)
            for job in as_completed(jobs):
                subject,i=jobs[job]
                try:
                    result=job.result();updates[subject]['added_ll'][i:i+len(result)]=result
                except Exception as exc:errors[subject]=f'{type(exc).__name__}: {exc}'
        next_pending=[]
        for subject,update in updates.items():
            record=fit[subject]
            if subject in errors:
                record['local_refinement_stopped']=errors[subject];continue
            old=localization.posterior_grid(base,'W',update['old_grid'],update['old_ll'])
            grid=np.concatenate((update['old_grid'],update['added_grid']));ll=np.concatenate((update['old_ll'],update['added_ll']))
            order=np.argsort(grid);grid,ll=grid[order],ll[order]
            new=localization.posterior_grid(base,'W',grid,ll)
            difference=float(max(abs(new['cdf']-np.interp(grid,update['old_grid'],old['cdf']))))
            record.update(grid=grid,parameter_log_likelihood=ll,grid_cdf_difference=difference,
                          posterior=localization.posterior_summary(base,'W',grid,ll,0.))
            for key in ('rank_u','covered95'):record['posterior'].pop(key,None)
            record['local_refinement_history'].append(dict(iteration=iteration+1,points=len(grid),new_points=len(update['added_grid']),cdf_difference=difference))
            if difference>refinement['maximum_cdf_difference']:next_pending.append(subject)
        print(f'b local grid round {iteration+1}: {len(next_pending)} still refining',flush=True)
        pending=next_pending
    # Independently check the complete refined curve at order 17.
    selected=[base['measured']['subjects'][i] for i in spec['posterior']['quadrature_check_subject_positions']]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        jobs={};values={}
        for subject in selected:
            if subject not in fit:continue
            record=fit[subject];grid=np.array(record['grid']);values[subject]=np.full(len(grid),np.nan)
            for i in range(0,len(grid),4):
                jobs[pool.submit(measured_likelihood_chunk,prepared[subject]['config'],prepared[subject]['arrays']['training'],grid[i:i+4],base['inference']['quadrature_check_order'])]=(subject,i)
        for job in as_completed(jobs):
            subject,i=jobs[job]
            try:
                result=job.result();values[subject][i:i+len(result)]=result
            except Exception as exc:fit[subject]['local_refinement_stopped']=f'order17: {type(exc).__name__}: {exc}'
        for subject,ll in values.items():
            if not np.isfinite(ll).all():fit[subject]['quadrature_cdf_difference']=float('inf');continue
            record=fit[subject];grid=record['grid']
            a=localization.posterior_grid(base,'W',grid,record['parameter_log_likelihood'])
            b=localization.posterior_grid(base,'W',grid,ll)
            record['quadrature_cdf_difference']=float(max(abs(a['cdf']-b['cdf'])))
            record['refined_quadrature_log_likelihood']=ll
    for subject,record in fit.items():write_json(Path(run_dir)/f'posterior_{subject}.json',record)
    return fit


def measured_observation_metrics(y,estimate,time_mask,columns,config,normalization):
    result={};names=('EEG','HbO','HbR');nu=config['model']['student_nu']
    for j in columns:
        error=estimate['clean_mean'][time_mask,j]-y[time_mask,j]
        variance=estimate['clean_variance'][time_mask,j]+config['model']['observation_scale'][j]**2*nu/(nu-2)
        result[names[j]]=dict(rmse=float(np.sqrt(np.mean(error**2))),
                             nrmse=float(np.sqrt(np.mean(error**2))/normalization[j]),
                             observation_coverage95=float(np.mean(abs(error)<=norm.ppf(.975)*np.sqrt(variance))),
                             mean_predictive_variance=float(variance.mean()),
                             mean_teacher_variance=float(estimate['clean_variance'][time_mask,j].mean()))
    return result


def measured_candidate_trial(base,spec,subject,position,detail,arrays,config,fit,candidate):
    y=arrays['targets'][position];template=arrays['templates'][position];donor=arrays['donors'][position]
    normalization=np.maximum(np.std(arrays['training'].reshape(-1,3),axis=0),1e-12)
    identity=[r for r in detail['trial_inventory'] if r['heldout']][position]
    payload={'observations':y}
    distribution=None if candidate=='U0_FIXED' else fit
    full=mixture_state(y,config,'W',distribution)
    result=dict(physical_pass=full['physical_pass'],stability=cross_parameter_stability(full),masks={})
    if distribution is not None and position==spec['posterior']['mixture_check_heldout_position']:
        refined=mixture_state(y,config,'W',distribution,spec['posterior']['mixture_check_points'])
        result['mixture_mean_difference_in_noise_units']=float(np.max(abs(full['clean_mean']-refined['clean_mean'])/np.array(config['model']['observation_scale'])))
        result['mixture_relative_variance_difference']=float(np.max(abs(full['clean_variance']-refined['clean_variance'])/np.maximum(refined['clean_variance'],1e-12)))
        grid=np.array(fit['grid']);ll=np.array(fit['parameter_log_likelihood']);lo,hi=base['axes']['W']['bounds'];band=(hi-lo)*spec['posterior']['boundary_band_fraction']
        restricted=dict(grid=grid,parameter_log_likelihood=np.where((grid>=lo+band)&(grid<=hi-band),ll,-np.inf))
        trimmed=mixture_state(y,config,'W',restricted)
        driver=full['state_mean'][:,0];other=trimmed['state_mean'][:,0]
        result['boundary_restricted_driver_correlation']=float(np.corrcoef(driver,other)[0,1])
        result['boundary_restricted_driver_nrmse']=float(np.sqrt(np.mean((driver-other)**2))/max(np.std(driver),1e-12))
    for mask_index,mask_name in enumerate(base['teacher']['masks']):
        visible=arrays['masked_inputs'][position,mask_index]
        _,time_mask,columns=masked_input(y,mask_name,base['inference']['mask_center_steps'])
        other=[j for j in range(3) if j not in columns]
        estimate=mixture_state(visible,config,'W',distribution)
        score=predictive_log_score(y,estimate,time_mask,columns,config)
        item=dict(joint_score=score,metrics=measured_observation_metrics(y,estimate,time_mask,columns,config,normalization),controls={},
                  physical_pass=estimate['physical_pass'],paired_variance_change=np.mean(estimate['clean_variance'][time_mask]-full['clean_variance'][time_mask],axis=0))
        if mask_name in base['teacher']['primary_masks']:
            for control in base['teacher']['controls']:
                null_input=visible.copy()
                if control=='own_history':null_input[:,other]=np.nan
                elif control=='own_history_and_task':null_input-=template;null_input[:,other]=np.nan
                elif control=='independent_pairing':null_input[:,other]=donor[:,other]
                elif control=='circular_shift':null_input[:,other]=np.roll(visible[:,other],round(len(y)*spec['controls']['circular_shift_fraction']),axis=0)
                null=mixture_state(null_input,config,'W',distribution)
                if control=='own_history_and_task':
                    null['clean_mean']+=template;null['component_clean_means']+=template[None]
                control_score=predictive_log_score(y,null,time_mask,columns,config)
                item['controls'][control]=dict(score=control_score,joint_increment=score-control_score,physical_pass=null['physical_pass'])
        else:
            prior=mixture_state(np.full_like(y,np.nan),config,'W',distribution)
            item['all_missing_prior_increment']=score-predictive_log_score(y,prior,time_mask,columns,config)
        result['masks'][mask_name]=item
        payload[f'{candidate}_{mask_name}_mean']=estimate['clean_mean'];payload[f'{candidate}_{mask_name}_variance']=estimate['clean_variance']
        payload[f'{candidate}_{mask_name}_mask']=np.isfinite(visible)
    result['physical_pass']=bool(result['physical_pass'] and all(m['physical_pass'] and all(v['physical_pass'] for v in m['controls'].values()) for m in result['masks'].values()))
    payload[f'{candidate}_full_mean']=full['clean_mean'];payload[f'{candidate}_full_variance']=full['clean_variance']
    payload[f'{candidate}_conditional_variance']=full['conditional_clean_variance'];payload[f'{candidate}_parameter_variance']=full['parameter_clean_variance']
    payload[f'{candidate}_driver_mean']=full['state_mean'][:,0];payload[f'{candidate}_driver_variance']=full['state_variance'][:,0]
    return result,payload


def measured_trial_case(base,spec,subject,position,detail,arrays,config,fit):
    identity=[r for r in detail['trial_inventory'] if r['heldout']][position]
    row=dict(subject=subject,heldout_position=position,identity=identity,execution='completed',candidates={})
    payload={'observations':arrays['targets'][position]}
    for candidate in spec['candidates']:
        try:
            result,values=measured_candidate_trial(base,spec,subject,position,detail,arrays,config,fit,candidate)
            result['execution']='completed';payload.update(values)
        except Exception as exc:
            result=dict(execution='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            row['execution']='partial'
        row['candidates'][candidate]=result
    return row,payload


def measured_trial_job(*args):
    try:return measured_trial_case(*args)
    except Exception as exc:
        return dict(subject=args[2],heldout_position=args[3],execution='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()),None


def measured_summary(base,spec,rows,fit,preparation_failures,metadata_summary):
    from collections import Counter
    def candidate_successes(candidate):
        return [r for r in rows if r.get('candidates',{}).get(candidate,{}).get('execution')=='completed']
    successes=candidate_successes(spec['primary'])
    successful_keys={(r['subject'],r['heldout_position']) for r in successes}
    failures=[r for r in rows if (r['subject'],r['heldout_position']) not in successful_keys]
    counts=Counter(r['subject'] for r in successes)
    complete_subjects=[s for s in base['measured']['subjects'] if counts[s]==spec['qualification']['complete_heldout_trials_per_subject']]
    summary=dict(stage='b',execution='completed',registered_heldout_trials=108,successful_heldout_trials=len(successes),
                 failed_heldout_trials=len(failures),case_failures=failures,preparation_failures=preparation_failures,
                 complete_subjects=complete_subjects,metadata_boundary=metadata_summary,candidates={},
                 estimand=base['measured']['estimand'],latent_ground_truth_available=False,
                 interval_kind=spec['scoring']['observation_interval'])
    if not complete_subjects:
        summary.update(qualified_candidates=[],uq_stage_eligible=False)
        return summary
    def subject_stat(function):
        return cluster_summary([np.mean([function(r) for r in successes if r['subject']==subject]) for subject in complete_subjects],base,spec['seed']+31)
    for candidate in spec['candidates']:
        successes=candidate_successes(candidate)
        successful_keys={(r['subject'],r['heldout_position']) for r in successes}
        candidate_counts=Counter(r['subject'] for r in successes)
        complete_subjects=[s for s in base['measured']['subjects'] if candidate_counts[s]==6]
        result=dict(masks={},checks={},successful_trials=len(successes),complete_subjects=complete_subjects,
                    failures=[dict(subject=r['subject'],heldout_position=r['heldout_position'],detail=r.get('candidates',{}).get(candidate,r)) for r in rows if (r['subject'],r['heldout_position']) not in successful_keys])
        if not complete_subjects:
            result['checks']['complete_registered_subjects_and_trials']=False
            result['measured_core_teacher_qualified']=False;summary['candidates'][candidate]=result;continue
        for mask in base['teacher']['masks']:
            names=('EEG',) if mask.endswith('EEG') else ('HbO','HbR')
            m=dict(joint_score=subject_stat(lambda r:r['candidates'][candidate]['masks'][mask]['joint_score']),metrics={},controls={})
            for name in names:
                m['metrics'][name]={metric:subject_stat(lambda r:r['candidates'][candidate]['masks'][mask]['metrics'][name][metric])
                                    for metric in ('nrmse','rmse','observation_coverage95','mean_predictive_variance','mean_teacher_variance')}
            if mask in base['teacher']['primary_masks']:
                for control in base['teacher']['controls']:
                    values=np.array([np.mean([r['candidates'][candidate]['masks'][mask]['controls'][control]['joint_increment']
                                             for r in successes if r['subject']==subject]) for subject in complete_subjects])
                    m['controls'][control]=cluster_summary(values,base,spec['seed']+32)
                    m['controls'][control]['leave_one_subject_out_mean_min']=float(min((values.sum()-v)/(len(values)-1) for v in values)) if len(values)>1 else None
            else:m['all_missing_prior_increment']=subject_stat(lambda r:r['candidates'][candidate]['masks'][mask]['all_missing_prior_increment'])
            result['masks'][mask]=m
        checks=result['checks'];q=spec['qualification'];all_entries=[r['candidates'][candidate] for r in successes]
        checks['complete_registered_subjects_and_trials']=len(complete_subjects)==q['complete_subjects'] and len(successes)==108 and not preparation_failures
        checks['physical']=all(e['physical_pass'] for e in all_entries)
        controls=[v for mask in base['teacher']['primary_masks'] for v in result['masks'][mask]['controls'].values()]
        checks['shared_information']=all(v['ci95'][0]>0 for v in controls)
        checks['not_driven_by_one_subject']=all(v['leave_one_subject_out_mean_min'] is not None and v['leave_one_subject_out_mean_min']>0 for v in controls)
        checks['masked_observation_accuracy']=all(m['nrmse']['mean']<=q['maximum_center_mask_observation_nrmse']
                                                 for mask in base['teacher']['primary_masks'] for m in result['masks'][mask]['metrics'].values())
        checks['synthetic_teacher_eligible']=candidate==spec['primary']
        if candidate==spec['primary']:
            checks['training_posterior_resolution']=len(fit)==18 and all(v['grid_cdf_difference']<=spec['posterior']['maximum_cdf_difference'] and
                    (v['quadrature_cdf_difference'] is None or v['quadrature_cdf_difference']<=spec['posterior']['maximum_quadrature_cdf_difference']) for v in fit.values())
            checked=[r['candidates'][candidate] for r in successes if r['heldout_position']==0]
            result['numerical_sensitivity']={key:([e[key] for e in checked]) for key in ('mixture_mean_difference_in_noise_units','mixture_relative_variance_difference','boundary_restricted_driver_correlation','boundary_restricted_driver_nrmse')}
            checks['parameter_mixture_resolution']=len(checked)==18 and all(e['mixture_mean_difference_in_noise_units']<=q['maximum_mixture_mean_difference_in_noise_units'] and
                     e['mixture_relative_variance_difference']<=q['maximum_mixture_relative_variance_difference'] for e in checked)
            checks['boundary_insensitive_driver']=len(checked)==18 and all(e['boundary_restricted_driver_correlation']>=q['minimum_boundary_restricted_driver_correlation'] and
                     e['boundary_restricted_driver_nrmse']<=q['maximum_boundary_restricted_driver_nrmse'] for e in checked)
        result['measured_core_teacher_qualified']=all(checks.values());summary['candidates'][candidate]=result
    summary['qualified_candidates']=[c for c,v in summary['candidates'].items() if v['measured_core_teacher_qualified']]
    summary['uq_stage_eligible']=bool(summary['qualified_candidates'])
    primary_rows=candidate_successes(spec['primary']);fixed_rows=candidate_successes('U0_FIXED')
    common={(r['subject'],r['heldout_position']) for r in fixed_rows}
    successes=[r for r in primary_rows if (r['subject'],r['heldout_position']) in common]
    counts=Counter(r['subject'] for r in successes);complete_subjects=[s for s in base['measured']['subjects'] if counts[s]==6]
    summary['primary_minus_fixed_score']={mask:subject_stat(lambda r:r['candidates'][spec['primary']]['masks'][mask]['joint_score']-r['candidates']['U0_FIXED']['masks'][mask]['joint_score']) for mask in base['teacher']['masks']} if complete_subjects else {}
    return summary


def measured_report(base,spec,summary):
    lines=['# Step5B 阶段报告：已知被试与 session 中的新 trial','',
           f"注册 18 被试 × 6 留出 trial = 108；主候选完成 {summary['successful_heldout_trials']}，失败 {summary['failed_heldout_trials']}；主候选具有完整六条留出结果的被试 {len(summary['complete_subjects'])}。",'',
           '训练范围为 subjects 01–18、sessions 01/03/05，每 session 8 train / 2 heldout。此处不是整 session 留出，也不是新被试泛化。',
           '输入和评分目标使用不同原始 trial 处理路径，隐藏值在滤波、功率和重采样前移除。PCA、通道选择、共同 HbO/HbR 比例、噪声尺度及参数后验仅使用训练 trial；留出 trial 不更新参数权重。',
           'own_history 对照使用同模态的未遮挡上下文，包括未来上下文；所有结果属于 fixed-interval 遮挡重建，不是因果预测。HbO/HbR 是显式近似 MBLL 后的共同缩放坐标，不是个体绝对生理浓度。','',
           '| 候选 | measured 核心资格 | 未通过检查 |','|---|---|---|']
    for c,v in summary['candidates'].items():lines.append(f"| {c} | {v['measured_core_teacher_qualified']} | {', '.join(k for k,x in v['checks'].items() if not x) or '无'} |")
    lines+=['','| 候选 | 遮挡 | 对照 | log-score 增量 | subject bootstrap 95% CI | 最小留一被试均值 |','|---|---|---|---:|---|---:|']
    for c,v in summary['candidates'].items():
        if not v['masks']:continue
        for mask in base['teacher']['primary_masks']:
            for control,s in v['masks'][mask]['controls'].items():
                loo=s['leave_one_subject_out_mean_min']
                lines.append(f"| {c} | {mask} | {control} | {s['mean']:.6f} | [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}] | {loo} |")
    lines+=['','| 候选 | 遮挡目标 | NRMSE（训练坐标SD单位） | 带噪观测95%覆盖 |','|---|---|---:|---:|']
    for c,v in summary['candidates'].items():
        if not v['masks']:continue
        for mask in base['teacher']['primary_masks']:
            for name,m in v['masks'][mask]['metrics'].items():lines.append(f"| {c} | {mask}/{name} | {m['nrmse']['mean']:.4f} | {m['observation_coverage95']['mean']:.4f} |")
    lines+=['','覆盖率使用 Gaussian-moment 95% 预测区间，仅为带噪观测诊断；没有真实 r 或 clean trajectory，不能声称潜在状态覆盖已验证。先按 trial，再按 subject 汇总，未把自相关时间点当独立样本。',
            '完整 posterior 边界/网格、参数混合加密、去边界敏感性、整模态缺失及配对方差变化保留在 posterior/case/summary JSON。固定模型仅作基线，不因 measured 某个分数较好而跳过其 synthetic 敏感性失败。','',
            f"可进入全面 UQ 的核心 teacher：**{', '.join(summary['qualified_candidates']) or '无'}**。",'',
            'subjects 19–23 没有作为新确认样本使用；subjects 24–29 未读取。没有 tokenizer 训练、外部发布或 protected evaluation。','']
    return '\n'.join(lines)


def run_measured(config_path,run_dir,teacher_dir,workers=None,reuse_dir=None):
    base,spec,metadata=load_measured_config(config_path);workers=workers or spec['workers']
    teacher_dir=Path(teacher_dir).resolve();teacher_manifest,_=require_measured_teacher(teacher_dir,base,spec)
    calibration_dir=ROOT/teacher_manifest['calibration_run']
    # This check uses retained synthetic innovations and precedes metadata/arrays.
    noise_check=noise_estimator_preflight(base,spec,calibration_dir)
    if not noise_check['passed']:raise ValueError('known-noise estimator preflight failed; no measured arrays opened')
    reuse_manifest=None;reuse_summary=None;exact_reuse=False
    if reuse_dir:
        reuse_dir=Path(reuse_dir).resolve();reuse_manifest=json.loads((reuse_dir/'manifest.json').read_text())
        reuse_summary=json.loads((reuse_dir/'summary.json').read_text())
        old=yaml.safe_load((reuse_dir/'resolved_measured_config.yaml').read_text());compatible=copy.deepcopy(spec)
        exact_reuse=old==spec
        compatible['schema']=old['schema']
        for name in ('fnirs_pair_validity','invalid_heldout_selected_pair'):compatible['preprocessing'].pop(name,None)
        compatible['posterior'].pop('local_refinement',None)
        if (not exact_reuse and compatible!=old) or reuse_manifest['execution']!='completed' or reuse_manifest['stage']!='b':
            raise ValueError('reuse is limited to the v1 input-support correction and same-data numerical refinement')
        if yaml.safe_load((reuse_dir/'resolved_config.yaml').read_text())!=base:
            raise ValueError('reuse base model differs')
        if exact_reuse:
            import ast
            before=ast.parse((reuse_dir/'runner_snapshot.py').read_text());after=ast.parse(Path(__file__).read_text())
            for name in ('measured_candidate_trial','measured_trial_case','measured_observation_metrics',
                         'mixture_state','measured_likelihood_chunk','refine_measured_posterior_mass'):
                nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for tree in (before,after)]
                if ast.dump(nodes[0])!=ast.dump(nodes[1]):raise ValueError(f'exact reuse scientific function changed: {name}')
    run_dir,manifest=start_stage(base,'b',run_dir);started=time.time()
    (run_dir/'resolved_measured_config.yaml').write_text(yaml.safe_dump(spec,sort_keys=False))
    manifest.update(measured_config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                    teacher_run=str(teacher_dir.relative_to(ROOT)),teacher_summary_sha256=hashlib.sha256((teacher_dir/'summary.json').read_bytes()).hexdigest(),
                    calibration_run=str(calibration_dir.relative_to(ROOT)),workers=workers,
                    documented_dataset='data/EEG+NIRS Single-Trial/Open access dataset for simultaneous EEG and NIRS Brain-Computer Interfaces (BCIs).html')
    write_json(run_dir/'noise_estimator_preflight.json',noise_check)
    prepared={};preparation_failures={};rows=[];reused_fit={};reuse_hashes={}
    try:
        from experiments.evaluate_t3_multisession_loso import _validate_metadata
        metadata_summary,inventory,hashes=_validate_metadata(metadata)
        write_json(run_dir/'metadata_boundary.json',dict(summary=metadata_summary,inventory=inventory,hashes=hashes))
        if reuse_dir:
            import shutil
            if hashes!=json.loads((reuse_dir/'metadata_boundary.json').read_text())['hashes']:
                raise ValueError('source metadata changed during reuse')
            for subject in base['measured']['subjects']:
                path=reuse_dir/f'prepared_{subject}.json'
                posterior_path=reuse_dir/f'posterior_{subject}.json'
                if exact_reuse and posterior_path.exists():
                    reused_fit[subject]=json.loads(posterior_path.read_text())
                    reuse_hashes[str(posterior_path.relative_to(ROOT))]=hashlib.sha256(posterior_path.read_bytes()).hexdigest()
                if not path.exists():continue
                detail=json.loads(path.read_text())
                with np.load(reuse_dir/f'prepared_{subject}.npz') as payload:arrays={k:payload[k] for k in payload.files}
                config=copy.deepcopy(base);config['model']['steps']=120;config['model']['observation_scale']=detail['observation_scale']
                prepared[subject]=dict(detail=detail,arrays=arrays,config=config)
                for suffix in ('json','npz'):
                    path=reuse_dir/f'prepared_{subject}.{suffix}'
                    shutil.copyfile(path,run_dir/path.name);reuse_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                path=reuse_dir/f'posterior_{subject}.json'
                if path.exists():
                    reused_fit[subject]=json.loads(path.read_text());reuse_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                elif not exact_reuse:preparation_failures[subject]=reuse_summary['preparation_failures'][subject]
            manifest.update(reuse_run=str(reuse_dir.relative_to(ROOT)),reused_prepared_subjects=sorted(prepared),
                            reuse_inputs_sha256=reuse_hashes,exact_configuration_reuse=exact_reuse,
                            case_predictions_recomputed_after_local_refinement=not exact_reuse)
        new_subjects=[s for s in base['measured']['subjects'] if s not in prepared]
        # The exact selected subject/record inventory has passed before dispatch.
        manifest['measured_data_read']=True;write_json(run_dir/'manifest.json',manifest)
        with ProcessPoolExecutor(max_workers=min(workers,4),mp_context=get_context('spawn')) as pool:
            jobs={pool.submit(prepare_measured_subject,base,spec,metadata,subject,noise_check['difference_mad_constant']):subject for subject in new_subjects}
            for job in as_completed(jobs):
                subject=jobs[job]
                try:
                    detail,arrays,config=job.result()
                    write_json(run_dir/f'prepared_{subject}.json',detail);np.savez_compressed(run_dir/f'prepared_{subject}.npz',**arrays)
                    prepared[subject]=dict(detail=detail,arrays=arrays,config=config)
                    print(f'b prepared {subject}',flush=True)
                except Exception as exc:
                    preparation_failures[subject]=dict(subject=subject,execution='failed',stage='native_preparation',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                    write_json(run_dir/f'failure_{subject}.json',preparation_failures[subject]);print(f'b preparation failed {subject}',flush=True)
        fit,fit_failures=fit_measured_posteriors(base,spec,{s:v for s,v in prepared.items() if s not in reused_fit},workers,run_dir)
        if exact_reuse:
            manifest['reconstructed_input_likelihood_checks']={}
            for subject in new_subjects:
                if subject not in reused_fit or subject not in prepared:continue
                record=reused_fit[subject];indices=[0,len(record['grid'])//2,len(record['grid'])-1]
                p=prepared[subject]
                values=measured_likelihood_chunk(p['config'],p['arrays']['training'],
                    np.array(record['grid'])[indices],base['inference']['quadrature_order'])
                old=np.array(record['parameter_log_likelihood'])[indices]
                if not np.allclose(values,old,rtol=1e-10,atol=1e-8):
                    raise ValueError('reconstructed inputs disagree with retained training likelihood')
                manifest['reconstructed_input_likelihood_checks'][subject]=dict(
                    points=np.array(record['grid'])[indices],maximum_absolute_difference=float(max(abs(np.array(values)-old))))
        preparation_failures.update(fit_failures)
        if 'local_refinement' in spec['posterior']:
            if not exact_reuse:fit.update(reused_fit)
            fit=refine_measured_posterior_mass(base,spec,prepared,fit,workers,run_dir) if fit else fit
        fit.update(reused_fit if exact_reuse or 'local_refinement' not in spec['posterior'] else {})
        for subject,record in fit.items():write_json(run_dir/f'posterior_{subject}.json',record)
        reused_cases=0
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            jobs={}
            for subject in base['measured']['subjects']:
                for position in range(6):
                    old_case=reuse_dir/f'case_{subject}_{position}.json' if exact_reuse else None
                    if (old_case is not None and old_case.exists() and subject not in new_subjects
                            and subject in reused_fit and subject not in preparation_failures):
                        row=json.loads(old_case.read_text());rows.append(row);reused_cases+=1
                        for path in (old_case,old_case.with_suffix('.npz')):
                            if path.exists():
                                shutil.copyfile(path,run_dir/path.name)
                                reuse_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                        continue
                    if subject in preparation_failures:
                        row=dict(subject=subject,heldout_position=position,execution='failed',error='SUBJECT_PREPARATION_OR_TRAINING_FAILED')
                        rows.append(row);write_json(run_dir/f'case_{subject}_{position}.json',row)
                    else:
                        p=prepared[subject]
                        jobs[pool.submit(measured_trial_job,base,spec,subject,position,p['detail'],p['arrays'],p['config'],fit[subject])]=(subject,position)
            for job in as_completed(jobs):
                row,arrays=job.result();rows.append(row);stem=f"case_{row['subject']}_{row['heldout_position']}"
                write_json(run_dir/f'{stem}.json',row)
                if arrays is not None:np.savez_compressed(run_dir/f'{stem}.npz',**arrays)
                print(f"b {len(rows)}/108 {stem} {row['execution']}",flush=True)
        rows.sort(key=lambda r:(r['subject'],r['heldout_position']))
        summary=measured_summary(base,spec,rows,fit,preparation_failures,metadata_summary)
        write_json(run_dir/'summary.json',summary);(run_dir/'summary.md').write_text(measured_report(base,spec,summary))
        manifest.update(execution='completed',completed_heldout_case_outcomes=len(rows),measured_subjects=base['measured']['subjects'],
                        measured_sessions=base['measured']['sessions'],elapsed_seconds=time.time()-started,
                        reused_case_outcomes=reused_cases,recomputed_case_outcomes=len(rows)-reused_cases,
                        reuse_inputs_sha256=reuse_hashes)
    except BaseException as exc:
        manifest.update(execution='failed',error=f'{type(exc).__name__}: {exc}',completed_heldout_case_outcomes=len(rows),elapsed_seconds=time.time()-started)
        raise
    finally:write_json(run_dir/'manifest.json',manifest)
    return summary


if __name__=='__main__':raise SystemExit(main())
