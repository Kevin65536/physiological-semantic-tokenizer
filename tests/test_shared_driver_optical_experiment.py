"""Public fixtures only: optical target migration, identity gates and inheritance."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from experiments.scripts import evaluate_shared_driver_reconstruction as run

CONFIG=run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_optical_v1.yaml'

@pytest.fixture
def cfg():return run.read_fixed_roi_optical_config(CONFIG)


def add_group(parent,cfg,spec):
    val=[i for i in range(24) if i%4==spec['outer']]
    trials=[dict(subject=spec['subject'],session=cfg['sessions'][i//8],original_ma_trial_position=cfg['original_positions'][i%8],
        event_index=2*cfg['original_positions'][i%8]+1,fnirs_time_ms=10000+i*1000,
        native_start_samples=dict(fnirs=50+i*10),sample_id=spec['subject']+'/'+str(i)) for i in range(24)]
    meta=dict(spec,train=[i for i in range(24) if i not in val],validation=val,trials=trials)
    if spec['kind']=='measured':meta['fixed_roi']=deepcopy(cfg['fixed_roi'])
    record=dict(status='completed',spec=spec,metadata=meta,driver_prior_sd=.2,selected_tau={m:2. for m in cfg['methods']})
    p=parent/'prepared'/f"{spec['group']}.json";run.write_json(p,record);p.with_suffix('.npz').write_bytes(b'no array reads at metadata gate')
    for method in run.FIXED_ROI_OPTICAL_METHODS:
        for mode in cfg['modes']:
            cell=run.nonlinear_cell_id(spec,method,mode);d=parent/'cells'/cell
            run.write_json(d/'result.json',dict(cell=cell,rows=[dict(spec,method=method,mode=mode,trial=i,
                session=cfg['sessions'][i//8],status='failed_training',converged=False) for i in val]))
            (d/'trajectories.npz').write_bytes(b'failed validation trajectory retained')
    d=run.gain_training_directory(parent,spec,'trained_tau_logflow1')
    run.write_json(d/'selection.json',dict(spec=spec,method='trained_tau_logflow1',expected_starts=3,terminal_starts=3,
        successful_starts=0,status='failed_training',parameter_value=None))
    for i,tau in enumerate(cfg['tau_training']['start_values']):
        run.write_json(d/f'start_{i}'/'result.json',dict(spec=spec,method='trained_tau_logflow1',start_index=i,
            start_tau=tau,expected_trials=18,status='failed_numerical',amplitude_weight=0.,converged=False))
        (d/f'start_{i}'/'trajectories.npz').write_bytes(b'failed training retained')
    return record

@pytest.fixture
def parent(tmp_path,cfg):
    root=tmp_path/'project';p=root/cfg['parent_run'];p.mkdir(parents=True)
    old=run.read_fixed_roi_flow_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_flow_v1.yaml')
    (p/'resolved_config.yaml').write_text(yaml.safe_dump(old))
    run.write_json(p/'manifest.json',dict(experiment_id=old['experiment_id'],execution='completed',pilot=False,
        synthetic_terminal=True,measured_terminal=True,project_root=str(root.resolve())))
    for spec in run.nonlinear_group_specs(old,'synthetic'):add_group(p,old,spec)
    return root,p,old

@pytest.fixture
def audit(parent,cfg):
    root,p,old=parent;a=root/cfg['optical_audit_run'];a.mkdir(parents=True)
    ac=yaml.safe_load((run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_optical_motion_audit_v1.yaml').read_text())
    (a/'resolved_config.yaml').write_text(yaml.safe_dump(ac))
    run.write_json(a/'manifest.json',dict(experiment_id=ac['experiment_id'],execution='completed',result_status='completed',project_root=str(root.resolve())))
    run.write_json(a/'synthetic_summary.json',dict(status='completed'))
    rows=[];checks=[]
    for spec in run.nonlinear_group_specs(old,'measured'):
        record=add_group(p,old,spec)
        if spec['outer']==0:
            for i,t in enumerate(record['metadata']['trials']):
                rows.append(dict(identity=dict(t,trial_index=i,selected_labels=['AF7Fp1lowWL','AF7Fp1highWL'])))
                checks.extend(dict(sample_id=t['sample_id'],stage=s,matches=True) for s in ('od','hb_native','hb_4hz'))
    run.write_json(a/'measured_summary.json',dict(status='completed',windows=72,rows=rows,reproduction_checks=checks,current_pipeline_reproduced=True))
    return root,p,a,old


def test_check_only_no_arrays_no_measured_metadata(parent,cfg,monkeypatch,capsys):
    root,p,old=parent
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array read'))
    run.gain_prior_fit_main(SimpleNamespace(config=CONFIG,fixed_roi_optical_fit=True,project_root=root,check_only=True))
    d=json.loads(capsys.readouterr().out);assert (d['synthetic_groups'],d['measured_groups'],d['new_measured_training_starts'])==(12,24,72)
    assert not (root/cfg['optical_audit_run']).exists()

@pytest.mark.parametrize('key,value',[('parent_run','elsewhere'),('optical_audit_run','elsewhere'),('optical_pipelines',['current']),
    ('methods',['fixed_tau_logflow1']),('subjects',['subject_01']),('modes',['full'])])
def test_contract_rejects_scope(cfg,tmp_path,key,value):
    cfg[key]=value;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):run.read_fixed_roi_optical_config(p)


def test_bounds_and_flow_weight_strict(cfg,tmp_path):
    for section,key,value in [('tau_training','bounds',[.1,8.]),('flow_prior','weights',dict(fixed_tau_logflow1=0.,trained_tau_logflow1=1.))]:
        c=deepcopy(cfg);c[section][key]=value;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
        with pytest.raises(ValueError):run.read_fixed_roi_optical_config(p)

@pytest.mark.parametrize('mutation',['missing_check','failed_check','duplicate_row','event','session','position','label','split','manifest'])
def test_measured_gate_before_arrays(audit,cfg,monkeypatch,mutation):
    root,p,a,old=audit;s=json.loads((a/'measured_summary.json').read_text())
    if mutation=='missing_check':s['reproduction_checks'].pop()
    elif mutation=='failed_check':s['reproduction_checks'][0]['matches']=False
    elif mutation=='duplicate_row':s['rows'][1]=deepcopy(s['rows'][0])
    elif mutation in ('event','session','position','label'):
        key={'event':'event_index','session':'session','position':'original_ma_trial_position','label':'selected_labels'}[mutation]
        s['rows'][0]['identity'][key]='wrong'
    elif mutation=='split':
        f=p/'prepared/subject_01_o0.json';r=json.loads(f.read_text());r['metadata']['train']=[];run.write_json(f,r)
    else:
        m=json.loads((a/'manifest.json').read_text());m['execution']='running';run.write_json(a/'manifest.json',m)
    run.write_json(a/'measured_summary.json',s)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array read before gate'))
    with pytest.raises(ValueError):run.validate_optical_parent(cfg,root,'measured')


def test_synthetic_inherits_failed_evidence_without_fit(parent,cfg,tmp_path,monkeypatch):
    root,p,old=parent;out=tmp_path/'new';spec=run.optical_group_specs(cfg,'synthetic')[0]
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('inheritance should copy bytes'))
    r=run.prepare_fixed_roi_optical_group((cfg,out,root,spec,False));assert r['inheritance']['status']=='inherited_nonindependent'
    run.inherit_flow_training(cfg,root,out,spec,method='trained_tau_logflow1')
    for m in cfg['methods']:
        r=run.nonlinear_cell((cfg,out,spec,m,'full',False));assert all(x['status']=='failed_training' for x in r['rows'])
        cell=run.nonlinear_cell_id(spec,m,'full');assert (out/'cells'/cell/'trajectories.npz').read_bytes()==(p/'cells'/cell/'trajectories.npz').read_bytes()
    assert (out/'prepared'/f"{spec['group']}.npz").read_bytes()==(p/'prepared'/f"{spec['group']}.npz").read_bytes()


def populate_arrays(audit,cfg):
    root,p,a,old=audit;rng=np.random.default_rng(42);raw=rng.normal(size=(72,300,2));traces={k:raw.copy() for k in ['current','no_motion','mne_tddr']}
    traces['no_motion']+=np.linspace(0,.3,300)[None,:,None]
    traces['mne_tddr']*=.7
    for spec in run.nonlinear_group_specs(old,'measured'):
        idx=cfg['subjects'].index(spec['subject']);hb=raw[idx*24:(idx+1)*24]*2.;e=rng.normal(size=(24,120))
        target=np.array([run.view(x,h,cfg,'full')[0] for x,h in zip(e,hb)])
        f=p/'prepared'/f"{spec['group']}.json";r=json.loads(f.read_text());meta=r['metadata']
        meta.update(normalization_sd=[1.,1.,1.],projection=dict(fnirs_factor=2.,inverse_fnirs_factor=.5,
            measurement_scale=dict(fnirs_common=1.),computational_scale=dict(fnirs_common=1.)))
        meta['fixed_roi'].update(fnirs_factor=2.,frozen_observation_loading=.2,common_training_MAD=.1)
        run.write_json(f,r);np.savez(f.with_suffix('.npz'),feature_eeg=e,feature_fnirs=hb,target=target,normalizer=np.ones(3))
    np.savez(a/'measured_traces.npz',**{k+'__hb_native':v for k,v in traces.items()})
    return traces


def test_preparation_training_only_scale_exact_eeg_and_current_reproduction(audit,cfg,tmp_path):
    root,p,a,old=audit;traces=populate_arrays(audit,cfg);spec=run.optical_group_specs(cfg,'measured')[0]
    out=tmp_path/'one';r=run.prepare_fixed_roi_optical_group((cfg,out,root,spec,False))
    with np.load(out/'prepared'/f"{spec['group']}.npz") as z:one={k:z[k] for k in z.files}
    with np.load(p/'prepared/subject_01_o0.npz') as z:
        assert np.array_equal(one['feature_eeg'],z['feature_eeg']);assert np.array_equal(one['target'][:,:,0],z['target'][:,:,0])
    val=r['metadata']['validation'];traces['no_motion'][val]*=100
    np.savez(a/'measured_traces.npz',**{k+'__hb_native':v for k,v in traces.items()})
    out2=tmp_path/'two';r2=run.prepare_fixed_roi_optical_group((cfg,out2,root,spec,False))
    assert r['metadata']['optical_processing']['fnirs_factor']==r2['metadata']['optical_processing']['fnirs_factor']
    assert r['metadata']['normalization_sd']==r2['metadata']['normalization_sd']
    traces['current'][0,0,0]+=1.
    np.savez(a/'measured_traces.npz',**{k+'__hb_native':v for k,v in traces.items()})
    with pytest.raises(ValueError,match='reproduce'):run.prepare_fixed_roi_optical_group((cfg,tmp_path/'bad',root,spec,False))


def test_declared_denominators(cfg):
    syn=run.optical_group_specs(cfg,'synthetic');meas=run.optical_group_specs(cfg,'measured')
    assert len({s['group'] for s in syn+meas})==36
    assert len(syn)*2*3*6==432 and len(meas)*2*3*6==864 and len(meas)*3==72
    assert all(run.flow_options(cfg,m)==dict(flow_prior_weight=1.,flow_prior_log_sd=np.log(2.)) for m in cfg['methods'])


def test_training_routes_flow_tau_free_initial_and_train_only(cfg,tmp_path,monkeypatch):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.optical_group_specs(cfg,'measured')[0];record=add_group(tmp_path/'source',cfg,spec);calls=[]
    def fit(y,p,dt,**kw):
        assert y.shape==(18,120,3) and np.all(y==2.)
        assert kw.get('tie_total_hb_to_volume',False) is False
        assert kw['flow_prior_weight']==1. and kw['flow_prior_log_sd']==np.log(2.)
        assert kw['parameter_name']=='tau' and kw['parameter_bounds']==[.5,8.]
        assert kw['parameter_prior_mean'] is None and kw['parameter_prior_log_sd'] is None
        assert kw['record_trace'] is True and kw['max_evaluations']==3600 and kw['substeps']==4
        calls.append(y.copy());return dict(status='failed_numerical',converged=False,parameter_value=2.)
    monkeypatch.setattr(lib,'fit_nonlinear_shared_parameter',fit)
    for folder,value in [('a',3.),('b',1e9)]:
        out=tmp_path/folder;run.write_json(out/'prepared'/f"{spec['group']}.json",record)
        target=np.full((24,120,3),2.);target[record['metadata']['validation']]=value
        np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=np.ones(3))
        r=run.train_gain_prior_start((cfg,out,spec,'trained_tau_logflow1',1,2.));assert r['status']=='failed_numerical'
    np.testing.assert_array_equal(*calls)

@pytest.mark.parametrize('method',run.FIXED_ROI_OPTICAL_METHODS)
def test_validation_routes_flow_free_initial_and_selected_tau(cfg,tmp_path,monkeypatch,method):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.optical_group_specs(cfg,'measured')[0];record=add_group(tmp_path/'source',cfg,spec)
    run.write_json(tmp_path/'prepared'/f"{spec['group']}.json",record)
    np.savez(tmp_path/'prepared'/f"{spec['group']}.npz",target=np.zeros((24,120,3)),normalizer=np.ones(3),
        feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)))
    if method.startswith('trained'):run.write_json(run.gain_training_directory(tmp_path,spec,method)/'selection.json',dict(status='completed',parameter_value=3.))
    calls=[];state=np.tile([0.,0.,1.,1.02,1.03,.99],(120,1))
    monkeypatch.setattr(run,'build_shared_driver_design',lambda *a,**k:None)
    monkeypatch.setattr(run,'fit_shared_driver',lambda *a,**k:dict(driver=np.zeros(120),states=state))
    def fit(y,p,dt,**kw):
        assert p.free.tau==(3. if method.startswith('trained') else 2.)
        assert kw.get('tie_total_hb_to_volume',False) is False and kw['flow_prior_weight']==1.
        assert kw['flow_prior_log_sd']==np.log(2.) and kw['substeps']==4
        assert len(kw['starts'])==2;np.testing.assert_array_equal(kw['starts'][0]['initial_state'],state[0,1:])
        calls.append(y.copy());return dict(status='failed_numerical',converged=False,starts=[])
    monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',fit)
    r=run.nonlinear_cell((cfg,tmp_path,spec,method,'full',False));assert len(calls)==6
    assert all(row['flow_prior_weight']==1. for row in r['rows'])


def test_measured_requires_complete_inherited_synthetic_before_audit_metadata(parent,cfg,monkeypatch):
    root,p,old=parent
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('measured read before synthetic terminal'))
    args=SimpleNamespace(config=CONFIG,fixed_roi_optical_fit=True,check_only=False,pilot=False,phase='measured',workers=1,
        project_root=root,run_dir=root/cfg['output_root']/'fixture')
    with pytest.raises(ValueError,match='synthetic terminal'):run.gain_prior_fit_main(args)


def test_summary_counts_and_separates_pipelines(cfg,tmp_path):
    specs=run.optical_group_specs(cfg,'synthetic')+run.optical_group_specs(cfg,'measured')
    plan=[(s,m,v) for s in specs for m in cfg['methods'] for v in cfg['modes']]
    summary=run.summarize_fixed_roi_tau(tmp_path,plan,False)
    assert summary['expected_cells']==216 and summary['expected_rows']==1296
    assert summary['expected_new_training_starts']==72 and summary['expected_inherited_training_starts']==36
    assert summary['expected_inherited_baseline_rows']==432 and summary['expected_new_validation_rows']==864
    assert summary['inherited_baseline_rows']==0 and summary['new_validation_rows']==0
    assert len(summary['aggregates'])==18
    assert {a['optical_pipeline'] for a in summary['aggregates'] if a['kind']=='measured'}=={'no_motion','mne_tddr'}


def test_full_synthetic_phase_inherits_all_failures_without_optimizer_or_optical_reads(parent,cfg,monkeypatch,capsys):
    from src.inference import shared_driver_reconstruction as lib
    root,p,old=parent;out=root/cfg['output_root']/'optical_fixture'
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('synthetic inheritance array read'))
    def work(worker,payloads,*unused):
        for payload in payloads:yield payload,worker(payload),None
    monkeypatch.setattr(run,'bounded_nonlinear_work',work)
    monkeypatch.setattr(run,'train_gain_prior_start',lambda *a,**k:pytest.fail('synthetic inheritance refit'))
    args=SimpleNamespace(config=CONFIG,fixed_roi_optical_fit=True,check_only=False,pilot=False,phase='synthetic',workers=1,
        project_root=root,run_dir=out)
    run.gain_prior_fit_main(args)
    summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['execution']=='synthetic_terminal' and manifest['synthetic_terminal'] is True
    assert summary['terminal_cells']==72 and summary['observed_rows']==432 and summary['completed_rows']==0
    assert summary['inherited_baseline_rows']==432 and summary['new_validation_rows']==0
    assert summary['inherited_training_groups']==12 and summary['new_training_groups']==0
    assert not (root/cfg['optical_audit_run']).exists()

@pytest.mark.parametrize('bad',['shape','nan'])
def test_optical_trace_shape_finite_fail_closed(audit,cfg,tmp_path,bad):
    root,p,a,old=audit;traces=populate_arrays(audit,cfg)
    if bad=='shape':traces['mne_tddr']=traces['mne_tddr'][:71]
    else:traces['mne_tddr'][0,0,0]=np.nan
    np.savez(a/'measured_traces.npz',**{k+'__hb_native':v for k,v in traces.items()})
    with pytest.raises(ValueError,match='shape/finite'):
        run.prepare_fixed_roi_optical_group((cfg,tmp_path/'out',root,run.optical_group_specs(cfg,'measured')[0],False))
