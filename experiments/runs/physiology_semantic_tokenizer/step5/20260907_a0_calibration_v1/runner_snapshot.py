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
    parser.add_argument('--stage',choices=['a0'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=None)
    args=parser.parse_args(); cfg=load_config(args.config)
    workers=args.workers or cfg['workers']
    run_calibration(cfg,args.run_dir,args.reference_run,workers)
    return 0


if __name__=='__main__':raise SystemExit(main())
