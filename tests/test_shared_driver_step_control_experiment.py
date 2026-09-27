"""Training-only numerical diagnostic: synthetic fixtures, no retained real arrays."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from experiments.scripts import evaluate_shared_driver_reconstruction as r
CONFIG=r.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_step_control_v1.yaml'
@pytest.fixture
def cfg():return r.read_conditional_step_config(CONFIG)
@pytest.fixture
def parent(tmp_path,cfg):
 root=tmp_path/'project';p=root/cfg['parent_run'];p.mkdir(parents=True)
 r.write_json(p/'manifest.json',dict(execution='completed',pilot=False,experiment_id=cfg['base_contract']['experiment_id'],project_root=str(root.resolve()),synthetic_terminal=True,measured_terminal=True))
 (p/'resolved_config.yaml').write_text(yaml.safe_dump(cfg['base_contract']))
 for spec in r.step_control_specs(cfg,'synthetic')+r.step_control_specs(cfg,'measured'):
  train=[i for i in range(24) if i%4!=spec['outer']];val=[i for i in range(24) if i%4==spec['outer']]
  meta=dict(train=train,validation=val,trials=[dict(session=cfg['base_contract']['sessions'][i//8],sample_id=str(i)) for i in range(24)])
  if spec['kind']=='measured':meta['optical_processing']=dict(pipeline=spec['optical_pipeline'])
  r.write_json(p/'prepared'/f"{spec['group']}.json",dict(status='completed',spec=spec,metadata=meta,conditional_mapping=dict(matrix=[[1.,0.],[0.,1.]],beta_reference=1.),driver_prior_sd=1.))
  for i,g in enumerate([.5,1.,2.]):
   r.write_json(r.gain_training_directory(p,spec,'conditional_trained_gain')/f'start_{i}'/'result.json',dict(spec=spec,start_index=i,start_relative_gain=g,training_trials=train,parameter_name='neurovascular_gain',status='completed' if i==1 else 'failed_numerical',tau=2.,parameter_prior_cost=0.,objective=10.+i,evaluations=100))
 return root,p

def arrays(p,spec):
 meta=json.loads((p/'prepared'/f"{spec['group']}.json").read_text())['metadata'];target=np.zeros((24,120,3));sd=np.ones(3)
 np.savez(p/'prepared'/f"{spec['group']}.npz",target=target,normalizer=sd)
 for i in range(3):np.savez(r.gain_training_directory(p,spec,'conditional_trained_gain')/f'start_{i}'/'trajectories.npz',target=target[meta['train']],normalizer=sd,trial_indices=meta['train'])

@pytest.mark.parametrize('field',['step_control','measured_groups','base_contract'])
def test_strict_contract(tmp_path,cfg,field):
 bad=deepcopy(cfg)
 if field=='base_contract':bad[field]['conditional_gain_training']['max_iterations']=91
 elif field=='measured_groups':bad[field]=bad[field]+['no_motion__subject_09_o0']
 else:bad[field]='armijo'
 p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(bad))
 with pytest.raises(ValueError):r.read_conditional_step_config(p)

def test_check_only_metadata_no_arrays(parent,cfg,monkeypatch,capsys):
 root,p=parent;monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array access'))
 r.conditional_step_control_main(SimpleNamespace(config=CONFIG,project_root=root,check_only=True))
 assert json.loads(capsys.readouterr().out)['source_arrays_read']==0
 assert len(r.step_control_specs(cfg,'synthetic'))==12 and len(r.step_control_specs(cfg,'measured'))==4

def test_copy_baseline_failures_exact_and_mismatch_rejected(parent,cfg,tmp_path):
 root,p=parent;spec=r.step_control_specs(cfg,'synthetic')[0];arrays(p,spec);out=tmp_path/'child'
 r.prepare_step_control_group(cfg,out,root,spec)
 assert (p/'prepared'/f"{spec['group']}.npz").read_bytes()==(out/'prepared'/f"{spec['group']}.npz").read_bytes()
 for i in range(3):
  src=r.gain_training_directory(p,spec,'conditional_trained_gain')/f'start_{i}'/'result.json';dest=out/'baseline_training'/f"{spec['group']}__conditional_trained_gain"/f'start_{i}'/'result.json'
  assert src.read_bytes()==dest.read_bytes()
 bad=r.gain_training_directory(p,spec,'conditional_trained_gain')/'start_0/trajectories.npz';np.savez(bad,target=np.zeros((18,120,3)),normalizer=np.ones(3)*2,trial_indices=list(range(18)))
 with pytest.raises(ValueError):r.prepare_step_control_group(cfg,tmp_path/'bad',root,spec)
 assert not (tmp_path/'bad/prepared').exists()

def test_scope_before_arrays(parent,cfg,monkeypatch,tmp_path):
 root,p=parent;s=deepcopy(r.step_control_specs(cfg,'measured')[0]);s['group']='no_motion__subject_09_o0'
 monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array access'))
 with pytest.raises(ValueError):r.prepare_step_control_group(cfg,tmp_path/'out',root,s)

def test_measured_gate_before_measured_metadata(parent,cfg,monkeypatch):
 root,p=parent;original=r.validate_step_parent;seen=[]
 def guard(c,rt,kind):
  seen.append(kind)
  assert kind=='synthetic'
  return original(c,rt,kind)
 monkeypatch.setattr(r,'validate_step_parent',guard);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array access'))
 args=SimpleNamespace(config=CONFIG,project_root=root,check_only=False,pilot=False,phase='measured',workers=1,run_dir=root/cfg['output_root']/'child')
 with pytest.raises(ValueError,match='synthetic recovery gate'):r.conditional_step_control_main(args)
 assert seen==['synthetic']

def test_worker_transmits_only_optional_step_preserves_training_objective(parent,cfg,tmp_path,monkeypatch):
 from src.inference import shared_driver_reconstruction as lib
 root,p=parent;spec=r.step_control_specs(cfg,'synthetic')[0];arrays(p,spec);out=tmp_path/'child';r.prepare_step_control_group(cfg,out,root,spec)
 captured={}
 def fit(target,params,dt,**kw):
  captured.update(kw);assert target.shape==(18,120,3) and params.free.tau==2.
  return dict(status='failed_numerical',converged=False,parameter_value=1.,relative_gain=1.)
 monkeypatch.setattr(lib,'fit_nonlinear_shared_parameter',fit)
 base=deepcopy(cfg['base_contract']);base['training_step_control']='quadratic_interpolation'
 record=r.train_gain_prior_start((base,str(out),spec,'conditional_trained_gain',1,1.))
 assert record['step_control']=='quadratic_interpolation' and captured['step_control']=='quadratic_interpolation'
 assert captured['max_evaluations']==3600 and captured['max_iterations']==90 and captured['gradient_tolerance']==1e-4
 assert captured['parameter_prior_mean'] is None and captured['parameter_prior_log_sd'] is None
 assert captured['penalty']==.01 and captured['initial_penalty']==100. and captured['flow_prior_weight']==1.
 np.testing.assert_array_equal(captured['starts'][0]['driver'],np.zeros((18,120)))
 np.testing.assert_array_equal(captured['starts'][0]['initial_state'],np.tile([0.,1.,1.,1.,1.],(18,1)))

@pytest.mark.parametrize('paired',[False,True])
def test_pilot_one_group_one_start_no_validation(parent,cfg,monkeypatch,paired):
 root,p=parent;spec=next(s for s in r.step_control_specs(cfg,'synthetic') if s['group']=='synthetic_g1_mixed_r0');arrays(p,spec);seen=[]
 def bounded(worker,payloads,*args):
  for payload in payloads:
   base,out,s,m,i,g=payload;seen.append((s['group'],i,g));result=dict(spec=s,status='completed',converged=True,expected_trials=18,completed_trials=18,parameter_value=1.,relative_gain=1.,beta_reference=1.,objective=1.,boundary_status='INTERIOR')
   r.write_json(r.gain_training_directory(out,s,m)/f'start_{i}'/'result.json',result);yield payload,result,None
 monkeypatch.setattr(r,'bounded_nonlinear_work',bounded)
 config=CONFIG.with_name('shared_driver_conditional_step_control_v2.yaml') if paired else CONFIG
 out=root/cfg['output_root']/'pilot';args=SimpleNamespace(config=config,project_root=root,check_only=False,pilot=True,phase='synthetic',workers=1,run_dir=out)
 r.conditional_step_control_main(args)
 assert seen==[('synthetic_g1_mixed_r0',1,1.)]*(2 if paired else 1) and not (out/'cells').exists()
 if paired:
  summary=json.loads((out/'synthetic_paired_summary.json').read_text())
  assert summary['expected_pairs']==summary['terminal_pairs']==1 and summary['numerical_screen_passed']
 else:
  summary=json.loads((out/'synthetic_training_summary.json').read_text());assert summary['expected_starts']==summary['terminal_starts']==1 and summary['engineering_screen']['all_groups_passed']
 assert json.loads((out/'manifest.json').read_text())['execution']=='completed'


@pytest.mark.parametrize('regression,missing',[(False,False),(True,False),(False,True)])
def test_paired_success_and_performance_require_complete_no_regression(tmp_path,cfg,monkeypatch,regression,missing):
 def summary(c,out,kind,pilot):
  candidate=out.name=='quadratic_interpolation'
  rows=[dict(group='group',start=0,status='failed_numerical' if candidate and regression else 'completed',
      converged=not(candidate and regression),optimization_evaluations=50 if candidate else 100)]
  if candidate and missing:rows=[]
  return dict(expected_starts=1,terminal_starts=len(rows),rows=rows,regressions=0,
      successful_starts=sum(x['converged'] for x in rows),engineering_screen=dict(all_groups_passed=True))
 monkeypatch.setattr(r,'summarize_step_control',summary)
 result=r.summarize_step_panel(cfg,tmp_path,'synthetic')
 assert result['numerical_screen_passed']==(not regression and not missing)
 assert result['performance_screen_passed']==(not regression and not missing)
 assert result['regressions']==int(regression)


def test_panel_covers_all_retained_subject_pipeline_folds_without_changing_objective(cfg):
 panel=r.read_conditional_step_config(CONFIG.with_name('shared_driver_conditional_step_control_v2.yaml'))
 assert panel['base_contract']==cfg['base_contract']
 specs=r.step_control_specs(panel,'measured')
 assert len(specs)==24 and {s['subject'] for s in specs}=={'subject_01','subject_09','subject_18'}
 assert {s['optical_pipeline'] for s in specs}=={'no_motion','mne_tddr'}


@pytest.mark.parametrize('bad_cost',[None,0,float('nan'),3601])
def test_missing_or_invalid_cost_cannot_be_counted_as_speedup(tmp_path,cfg,monkeypatch,bad_cost):
 def summary(c,out,kind,pilot):
  cost=bad_cost if out.name=='quadratic_interpolation' else 100
  return dict(expected_starts=1,terminal_starts=1,successful_starts=1,regressions=0,
      rows=[dict(group='group',start=0,status='completed',converged=True,optimization_evaluations=cost)])
 monkeypatch.setattr(r,'summarize_step_control',summary)
 result=r.summarize_step_panel(cfg,tmp_path,'measured')
 assert result['numerical_screen_passed']
 assert not result['performance_screen_passed'] and result['evaluation_reduction_fraction'] is None


@pytest.mark.parametrize('fault',[None,'target','gradient','integration','missing_pair'])
def test_report_audit_keeps_failures_and_rejects_invalid_comparisons(tmp_path,fault):
 from experiments.scripts.render_shared_driver_reconstruction import step_control_audit
 (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump(dict(schema='shared_driver_conditional_step_control_v2',synthetic_starts=1,measured_starts=1)))
 for kind in ('synthetic','measured'):
  r.write_json(tmp_path/f'{kind}_paired_summary.json',dict(complete=True,terminal_pairs=1,
      rows=[] if fault=='missing_pair' else [dict(group=kind,start=0)]))
  for arm in ('armijo','quadratic_interpolation'):
   failed=kind=='measured' and arm=='armijo';candidate=arm=='quadratic_interpolation'
   record=dict(spec=dict(subject='subject_01',optical_pipeline='no_motion'),
       status='failed_numerical' if failed else 'completed',converged=not failed,
       objective=1.,relative_gain=1.,parameter_value=1.,optimization_evaluations=3600 if failed else 100,
       seconds=10.,cpu_seconds=9.,projected_scaled_gradient_inf_norm=.1 if failed else float('nan') if fault=='gradient' and candidate else 1e-5,
       integration_checks=[] if fault=='integration' and candidate else [dict(difference_training_sd=.001)])
   folder=tmp_path/'arms'/arm/'training'/f'{kind}__conditional_trained_gain/start_0'
   r.write_json(folder/'result.json',record)
   target=np.zeros((18,3,3))+(1 if fault=='target' and candidate else 0)
   np.savez(folder/'trajectories.npz',target=target,normalizer=np.ones(3),trial_indices=np.arange(18),
       prediction=np.zeros((18,3,3)),driver=np.zeros((18,3)),states=np.ones((18,3,6)))
   if not candidate:r.write_json(tmp_path/'arms/armijo/baseline_training'/f'{kind}__conditional_trained_gain/start_0/result.json',record)
 if fault:
  with pytest.raises(ValueError):step_control_audit(tmp_path)
 else:
  frame,summary=step_control_audit(tmp_path)
  assert len(frame)==2 and frame.recovered.sum()==1 and frame.regressed.sum()==0
  assert summary.old_optimization_evaluations.sum()==3700 and frame.legacy_reproduced.all()
