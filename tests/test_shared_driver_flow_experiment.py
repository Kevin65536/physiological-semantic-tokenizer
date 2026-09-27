"""Small temporary evidence fixtures for the inherited fixed-ROI flow experiment."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as run

CONFIG = run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_flow_v1.yaml'


@pytest.fixture
def flow_cfg():
    return run.read_fixed_roi_flow_config(CONFIG)


@pytest.fixture
def parent(tmp_path,flow_cfg):
    root=tmp_path/'project';parent=root/flow_cfg['parent_run'];parent.mkdir(parents=True)
    cfg=run.read_fixed_roi_tau_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml')
    (parent/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    run.write_json(parent/'manifest.json',dict(execution='completed',pilot=False,
        experiment_id=cfg['experiment_id'],project_root=str(root.resolve()),synthetic_terminal=True,measured_terminal=True))
    return root,parent


def prepared_record(cfg,spec):
    validation=[i for i in range(24) if i%4==spec['outer']]
    train=[i for i in range(24) if i not in validation]
    trials=[dict(session=cfg['sessions'][i//8],sample_id=spec['group']+'/'+str(i),
        subject=spec['subject'],original_ma_trial_position=cfg['original_positions'][i%8]) for i in range(24)]
    meta=dict(spec,train=train,validation=validation,trials=trials)
    if spec['kind']=='measured':meta['fixed_roi']=cfg['fixed_roi']
    return dict(spec=spec,status='completed',metadata=meta,driver_prior_sd=.2,
                selected_tau={m:2. for m in run.FIXED_ROI_TAU_METHODS})


def add_prepared(parent,cfg,spec):
    record=prepared_record(cfg,spec);path=parent/'prepared'/f"{spec['group']}.json"
    run.write_json(path,record)
    arrays=dict(target=np.ones((24,120,3)),feature_eeg=np.zeros((24,120)),
        feature_fnirs=np.zeros((24,300,2)),normalizer=np.array([.2,.3,.4]))
    if spec['kind']=='synthetic':arrays.update(truth=np.ones((24,120)),truth_states=np.ones((24,120,6)))
    np.savez(path.with_suffix('.npz'),**arrays)
    return record


def add_phase_records(parent,cfg,kind):
    for spec in run.nonlinear_group_specs(cfg,kind):
        prep=add_prepared(parent,cfg,spec)
        for method in run.FIXED_ROI_FLOW_INHERITED:
            for mode in cfg['modes']:
                identifier=run.nonlinear_cell_id(spec,method,mode)
                dest=parent/'cells'/identifier/'result.json'
                rows=[dict(spec,method=method,mode=mode,trial=t,session=cfg['sessions'][t//8],
                    status='failed_training',converged=False,nrmse_EEG=None) for t in prep['metadata']['validation']]
                run.write_json(dest,dict(cell=identifier,spec=spec,method=method,mode=mode,status='completed',rows=rows))
                np.savez(dest.parent/'trajectories.npz',prediction=np.full((6,120,3),np.nan))
        directory=run.gain_training_directory(parent,spec,'fixed_roi_trained_tau')
        run.write_json(directory/'selection.json',dict(spec=spec,method='fixed_roi_trained_tau',expected_starts=3,
            terminal_starts=3,successful_starts=0,status='failed_training',parameter_value=None))
        for i,tau in enumerate(cfg['tau_training']['start_values']):
            dest=directory/f'start_{i}'/'result.json'
            run.write_json(dest,dict(spec=spec,method='fixed_roi_trained_tau',start_index=i,start_tau=tau,
                expected_trials=18,completed_trials=18,status='failed_numerical',converged=False))
            if i:np.savez(dest.parent/'trajectories.npz',last_valid=np.array([i]))


def test_flow_check_only_no_array_reads(parent,monkeypatch,capsys):
    root,_=parent
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('check-only opened arrays'))
    run.gain_prior_fit_main(SimpleNamespace(config=CONFIG,fixed_roi_flow_fit=True,check_only=True,project_root=root))
    info=json.loads(capsys.readouterr().out)
    assert info['source_arrays_read']==0 and info['training_starts_per_group']==9 and info['validation_rows_per_group']==108


@pytest.mark.parametrize('section,key,value',[
    (None,'parent_run','other'),(None,'methods',['fixed_tau_logflow1']),
    ('tau_training','bounds',[.1,100]),('tau_training','max_evaluations',7200),
    ('flow_prior','log_sd',.2),('flow_prior','weights',{'trained_tau_logflow1':1.}),
    ('fixed_roi','pair_index',1),('regularization','driver_amplitude_weight',1.),
])
def test_flow_config_strict(flow_cfg,tmp_path,section,key,value):
    bad=deepcopy(flow_cfg);(bad if section is None else bad[section])[key]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):run.read_fixed_roi_flow_config(path)


@pytest.mark.parametrize('change',['unfinished','pilot','identity','config','unknown'])
def test_flow_parent_fails_before_arrays(parent,flow_cfg,monkeypatch,change):
    root,directory=parent
    if change in ['unfinished','pilot','identity']:
        path=directory/'manifest.json';m=json.loads(path.read_text())
        m[{'unfinished':'execution','pilot':'pilot','identity':'experiment_id'}[change]]={'unfinished':'running','pilot':True,'identity':'OTHER'}[change]
        run.write_json(path,m)
    elif change=='unknown':flow_cfg['unknown']=True
    else:
        path=directory/'resolved_config.yaml';cfg=yaml.safe_load(path.read_text());cfg['tau_training']['bounds']=[.3,8.]
        path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('arrays before parent validation'))
    with pytest.raises(ValueError):run.validate_flow_parent(flow_cfg,root)


def test_phase_parent_audit_preserves_complete_failures_and_no_arrays(parent,flow_cfg,monkeypatch):
    root,directory=parent;add_phase_records(directory,flow_cfg,'synthetic')
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata audit opened arrays'))
    run.validate_flow_parent(flow_cfg,root,kind='synthetic')
    spec=run.nonlinear_group_specs(flow_cfg,'synthetic')[0]
    cell=directory/'cells'/run.nonlinear_cell_id(spec,'fixed_roi_trained_tau','full')/'result.json'
    record=json.loads(cell.read_text());record['rows'].pop();run.write_json(cell,record)
    with pytest.raises(ValueError,match='identities'):run.validate_flow_parent(flow_cfg,root,kind='synthetic')


@pytest.mark.parametrize('change',['split','identity','roi','phase'])
def test_prepared_rejects_before_array_read(parent,flow_cfg,tmp_path,monkeypatch,change):
    root,directory=parent;spec=run.nonlinear_group_specs(flow_cfg,'measured')[0]
    record=add_prepared(directory,flow_cfg,spec);out=tmp_path/'out'
    if change=='split':record['metadata']['train'][0]=record['metadata']['validation'][0]
    elif change=='identity':record['metadata']['trials'][0]['subject']='subject_09'
    elif change=='roi':record['metadata']['fixed_roi']=dict(record['metadata']['fixed_roi'],pair_index=1)
    else:run.write_json(out/'manifest.json',dict(phase='synthetic_preparation'))
    run.write_json(directory/'prepared'/f"{spec['group']}.json",record)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('invalid metadata opened arrays'))
    with pytest.raises(ValueError):run.prepare_fixed_roi_flow_group((flow_cfg,out,root,spec,False))


def test_prepared_is_exact_copy_without_measured_array_access(parent,flow_cfg,tmp_path,monkeypatch):
    root,directory=parent;spec=run.nonlinear_group_specs(flow_cfg,'synthetic')[0]
    old=add_prepared(directory,flow_cfg,spec);out=tmp_path/'out';source=directory/'prepared'/f"{spec['group']}.npz"
    original=source.read_bytes();loader=np.load;calls=[]
    def load(path,*a,**k):
        assert Path(path)==source
        calls.append(str(path));return loader(path,*a,**k)
    monkeypatch.setattr(np,'load',load)
    record=run.prepare_fixed_roi_flow_group((flow_cfg,out,root,spec,False))
    assert record['inheritance']['status']=='inherited_nonindependent'
    assert {k:v for k,v in record.items() if k!='inheritance'}==old
    assert source.read_bytes()==original==(out/'prepared'/source.name).read_bytes()
    assert len(calls)==1


def test_inheritance_preserves_failed_training_and_cells_without_fitting(parent,flow_cfg,tmp_path,monkeypatch):
    root,directory=parent;add_phase_records(directory,flow_cfg,'synthetic')
    spec=run.nonlinear_group_specs(flow_cfg,'synthetic')[0];out=tmp_path/'out'
    run.prepare_fixed_roi_flow_group((flow_cfg,out,root,spec,False))
    selection=run.inherit_flow_training(flow_cfg,root,out,spec)
    assert selection['status']=='failed_training' and selection['terminal_starts']==3
    for i in range(3):
        path=run.gain_training_directory(out,spec,'fixed_roi_trained_tau')/f'start_{i}'
        assert json.loads((path/'result.json').read_text())['status']=='failed_numerical'
        assert (path/'trajectories.npz').exists()==bool(i)
    from src.inference import shared_driver_reconstruction as lib
    monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',lambda *a,**k:pytest.fail('inherited baseline refitted'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('inherited cell need not open arrays'))
    result=run.nonlinear_cell((flow_cfg,out,spec,'fixed_roi_trained_tau','full',False))
    assert len(result['rows'])==6 and all(r['status']=='failed_training' for r in result['rows'])
    assert result['inheritance']['status']=='inherited_nonindependent'
    ident=run.nonlinear_cell_id(spec,'fixed_roi_trained_tau','full')
    assert (out/'cells'/ident/'trajectories.npz').read_bytes()==(directory/'cells'/ident/'trajectories.npz').read_bytes()


@pytest.mark.parametrize('method,weight', [('trained_tau_logflow01',.1),('trained_tau_logflow1',1.),('trained_tau_logflow4',4.)])
def test_training_flow_weight_passed_without_validation_leakage(flow_cfg,tmp_path,monkeypatch,method,weight):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.nonlinear_group_specs(flow_cfg,'synthetic')[0];record=prepared_record(flow_cfg,spec)
    state=np.tile([0.,0.,1.,1.,1.,1.],(120,1));calls=[]
    def fit(y,p,dt,**kw):
        assert y.shape==(18,120,3) and np.all(y==2.)
        calls.append(y.copy());assert kw['parameter_name']=='tau' and kw['parameter_prior_mean'] is None
        assert kw['parameter_prior_log_sd'] is None and kw['record_trace'] is True
        assert kw['flow_prior_weight']==weight and kw['flow_prior_log_sd']==np.log(2.)
        assert kw['driver_amplitude_weight']==0. and kw['max_evaluations']==3600
        assert kw['parameter_bounds']==[.5,8.] and kw['starts'][0]['parameter_value']==2.
        return dict(status='completed',converged=True,parameter_value=3.,objective=4.,expected_trials=18,completed_trials=18,
            flow_prior_cost=1.2,flow_prior_weight=weight,flow_prior_log_sd=np.log(2.),
            driver=np.zeros((18,120)),initial_state=np.tile(state[0,1:],(18,1)),states=np.tile(state,(18,1,1)),
            prediction=np.zeros((18,120,3)),canonical_prediction=np.zeros((18,120,3)),trace=[dict(objective=5.)])
    def replay(driver,initial,p,dt,**kw):
        assert p.free.tau==3. and kw['substeps']==8
        return dict(status='completed',states=state,canonical_prediction=np.zeros((120,3)))
    monkeypatch.setattr(lib,'fit_nonlinear_shared_parameter',fit);monkeypatch.setattr(run,'replay_nonlinear_driver',replay)
    for i,value in enumerate([3.,1e8]):
        out=tmp_path/str(i);run.write_json(out/'prepared'/f"{spec['group']}.json",record)
        target=np.full((24,120,3),2.);target[record['metadata']['validation']]=value
        np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=np.array([.2,.3,.4]))
        result=run.train_gain_prior_start((flow_cfg,out,spec,method,1,2.))
        assert result['status']=='completed' and result['flow_prior_cost']==1.2
    np.testing.assert_array_equal(*calls)


@pytest.mark.parametrize('method,weight,tau',[('fixed_tau_logflow1',1.,2.),('trained_tau_logflow01',.1,3.),
    ('trained_tau_logflow1',1.,3.),('trained_tau_logflow4',4.,3.)])
def test_validation_passes_flow_weight_and_frozen_tau(flow_cfg,tmp_path,monkeypatch,method,weight,tau):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.nonlinear_group_specs(flow_cfg,'measured')[0];record=add_prepared(tmp_path,flow_cfg,spec)
    if method.startswith('trained'):
        run.write_json(run.gain_training_directory(tmp_path,spec,method)/'selection.json',dict(status='completed',parameter_value=tau))
    state=np.tile([0.,0.,1.,1.,1.,1.],(120,1));calls=[]
    monkeypatch.setattr(run,'build_shared_driver_design',lambda *a,**k:None)
    monkeypatch.setattr(run,'fit_shared_driver',lambda *a,**k:dict(driver=np.zeros(120),states=state))
    def fit(y,p,dt,**kw):
        assert p.free.tau==tau and p.fixed.neurovascular_gain==1.
        assert kw['flow_prior_weight']==weight and kw['flow_prior_log_sd']==np.log(2.)
        calls.append(1);return dict(status='failed_domain',converged=False,starts=[],flow_prior_cost=2.3)
    monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',fit)
    result=run.nonlinear_cell((flow_cfg,tmp_path,spec,method,'center_EEG',False))
    assert len(calls)==6 and len(result['rows'])==6
    assert all(r['flow_prior_weight']==weight and r['flow_prior_cost']==2.3 for r in result['rows'])


def test_measured_waits_for_full_synthetic_before_arrays(parent,flow_cfg,monkeypatch):
    root,_=parent
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('measured accessed before synthetic gate'))
    args=SimpleNamespace(config=CONFIG,fixed_roi_flow_fit=True,check_only=False,pilot=False,phase='measured',workers=1,
        project_root=root,run_dir=root/flow_cfg['output_root']/'fixture')
    with pytest.raises(ValueError,match='synthetic terminal'):run.gain_prior_fit_main(args)


@pytest.mark.parametrize('phase,pilot',[('all',False),('measured',True)])
def test_flow_phase_contract(parent,phase,pilot):
    root,_=parent
    args=SimpleNamespace(config=CONFIG,fixed_roi_flow_fit=True,check_only=False,pilot=pilot,phase=phase,project_root=root)
    with pytest.raises(ValueError,match='synthetic'):run.gain_prior_fit_main(args)
