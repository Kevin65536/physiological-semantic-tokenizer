"""Array-free parent gates and synthetic fixtures for the initial-state contrast."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as run

CONFIG = run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_initial_v1.yaml'


@pytest.fixture
def cfg():
    return run.read_fixed_roi_initial_config(CONFIG)


def metadata(cfg,spec):
    validation=[i for i in range(24) if i%4==spec['outer']]
    meta=dict(spec,train=[i for i in range(24) if i not in validation],validation=validation,
        trials=[dict(subject=spec['subject'],sample_id=spec['group']+'/'+str(i),session=cfg['sessions'][i//8],
            original_ma_trial_position=cfg['original_positions'][i%8]) for i in range(24)])
    if spec['kind']=='measured':meta['fixed_roi']=deepcopy(cfg['fixed_roi'])
    return dict(status='completed',spec=spec,metadata=meta,driver_prior_sd=.2,
        selected_tau={m:2. for m in cfg['methods']})


def add_group(parent,cfg,spec,arrays=False):
    prep=metadata(cfg,spec);p=parent/'prepared'/f"{spec['group']}.json";run.write_json(p,prep)
    if arrays:
        np.savez(p.with_suffix('.npz'),target=np.ones((24,120,3)),normalizer=np.array([.2,.3,.4]),
            feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)))
    else:p.with_suffix('.npz').write_bytes(b'metadata audit must not open this fixture')
    for method in run.FIXED_ROI_INITIAL_FREE:
        for mode in cfg['modes']:
            identifier=run.nonlinear_cell_id(spec,method,mode);path=parent/'cells'/identifier/'result.json'
            run.write_json(path,dict(cell=identifier,spec=spec,method=method,mode=mode,status='completed',
                rows=[dict(spec,method=method,mode=mode,trial=i,session=cfg['sessions'][i//8],
                    status='failed_training',converged=False) for i in prep['metadata']['validation']]))
            (path.parent/'trajectories.npz').write_bytes(b'failed parent arrays retained unchanged')
    directory=run.gain_training_directory(parent,spec,'trained_tau_logflow1')
    run.write_json(directory/'selection.json',dict(spec=spec,method='trained_tau_logflow1',expected_starts=3,
        terminal_starts=3,successful_starts=0,status='failed_training',parameter_value=None))
    for i,tau in enumerate(cfg['tau_training']['start_values']):
        path=directory/f'start_{i}'/'result.json'
        run.write_json(path,dict(spec=spec,method='trained_tau_logflow1',start_index=i,start_tau=tau,
            expected_trials=18,completed_trials=18,status='failed_numerical',converged=False))
        (path.parent/'trajectories.npz').write_bytes(b'last-valid-parent-state')
    return prep


@pytest.fixture
def parent(tmp_path,cfg):
    root=tmp_path/'project';parent=root/cfg['parent_run'];parent.mkdir(parents=True)
    old=run.read_fixed_roi_flow_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_flow_v1.yaml')
    (parent/'resolved_config.yaml').write_text(yaml.safe_dump(old))
    run.write_json(parent/'manifest.json',dict(execution='completed',pilot=False,experiment_id=old['experiment_id'],
        project_root=str(root.resolve()),synthetic_terminal=True,measured_terminal=True))
    for spec in run.nonlinear_group_specs(old,'synthetic'):add_group(parent,old,spec)
    return root,parent,old


def test_check_only_verifies_36_groups_without_array_reads(parent,cfg,monkeypatch,capsys):
    root,_,_=parent
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('check-only array access'))
    run.gain_prior_fit_main(SimpleNamespace(config=CONFIG,fixed_roi_initial_fit=True,check_only=True,project_root=root))
    result=json.loads(capsys.readouterr().out)
    assert result['synthetic_groups']==36 and result['measured_groups']==12 and result['source_arrays_read']==0
    assert result['training_starts_per_group']==6 and result['new_measured_training_starts']==36
    groups=run.initial_group_specs(cfg,'synthetic')
    assert len({g['group'] for g in groups})==36
    assert {g['true_initial_log_p_over_v'] for g in groups}=={-.05,0.,.05}


@pytest.mark.parametrize('section,key,value',[(None,'parent_run','other'),(None,'methods',['fixed_tau_logflow1_tied']),
    ('synthetic','initial_log_p_over_v',[-.1,0.,.1]),('tau_training','bounds',[.1,100.]),
    ('initial_constraint','validation_start','project_free_linear_state'),('flow_prior','weights',{'trained_tau_logflow1_tied':4.})])
def test_initial_contract_rejects_changed_scope(cfg,tmp_path,section,key,value):
    bad=deepcopy(cfg);(bad if section is None else bad[section])[key]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):run.read_fixed_roi_initial_config(path)


@pytest.mark.parametrize('change',['unfinished','wrong_id','missing_synthetic_row','altered_split','unknown_config'])
def test_parent_metadata_fails_before_arrays(parent,cfg,monkeypatch,change):
    root,path,old=parent
    if change in ('unfinished','wrong_id'):
        m=json.loads((path/'manifest.json').read_text());m['execution' if change=='unfinished' else 'experiment_id']='invalid'
        run.write_json(path/'manifest.json',m)
    elif change=='unknown_config':cfg['extra']=True
    else:
        spec=run.nonlinear_group_specs(old,'synthetic')[0]
        if change=='missing_synthetic_row':
            file=path/'cells'/run.nonlinear_cell_id(spec,'fixed_tau_logflow1','full')/'result.json'
            r=json.loads(file.read_text());r['rows'].pop()
        else:
            file=path/'prepared'/f"{spec['group']}.json";r=json.loads(file.read_text());r['metadata']['train'][0]=0
        run.write_json(file,r)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('parent gate opened arrays'))
    with pytest.raises(ValueError):run.validate_initial_parent(cfg,root)


def test_synthetic_pairing_across_tau_and_initial_ratio(cfg,monkeypatch):
    def replay(r,initial,p,dt,**kw):
        decay=np.exp(-np.arange(len(r))*.05)
        canonical=np.column_stack((r,p.free.tau*r+(initial[3]-initial[2])*decay,
                                    .3*p.free.tau*r+(initial[4]-1)*decay))
        states=np.tile(np.r_[r[0],initial],(len(r),1));states[:,0]=r
        return dict(status='completed',canonical_prediction=canonical,states=states)
    monkeypatch.setattr(run,'replay_nonlinear_driver',replay)
    specs=[s for s in run.initial_group_specs(cfg,'synthetic') if s['spectrum']=='mixed' and s['replicate']==0]
    panels=[run.nonlinear_synthetic_panel(cfg,s)[0] for s in specs]
    op=run.native_feature_operators(120)['native_interpolation']
    standardized=[]
    for spec,panel in zip(specs,panels):
        initial=panel['truth_states'][:,0,1:]
        np.testing.assert_allclose(np.log(initial[:,3]/initial[:,2]),spec['true_initial_log_p_over_v'],atol=2e-16)
        np.testing.assert_array_equal(panel['truth'],panels[0]['truth'])
        np.testing.assert_array_equal(initial[:,[0,1,2,4]],panels[0]['truth_states'][:,0,1:][:,[0,1,2,4]])
        draws=[]
        for i in range(24):
            canonical=replay(panel['truth'][i],initial[i],run.parameters(cfg,spec['true_tau']),.25)['canonical_prediction']
            sd=.1*np.std(canonical,axis=0)
            draws.append(np.r_[(panel['feature_eeg'][i]-canonical[:,0])/sd[0],
                                ((panel['feature_fnirs'][i]-op@canonical[:,1:])/sd[1:]).ravel()])
        standardized.append(np.array(draws))
    for draws in standardized[1:]:np.testing.assert_allclose(draws,standardized[0],atol=2e-14,rtol=0.)


def test_synthetic_preparation_is_fresh_and_never_opens_parent_arrays(parent,cfg,tmp_path,monkeypatch):
    root,_,_=parent;spec=run.initial_group_specs(cfg,'synthetic')[0];out=tmp_path/'out';calls=[]
    def generate(config,identity):
        calls.append(identity)
        return dict(normalizer=np.ones(3),target=np.zeros((24,120,3))),metadata(config,identity)['metadata']
    monkeypatch.setattr(run,'nonlinear_synthetic_panel',generate)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('fresh synthetic read parent arrays'))
    result=run.prepare_fixed_roi_flow_group((cfg,out,root,spec,False))
    assert result['status']=='completed' and calls==[spec] and 'inheritance' not in result
    assert result['metadata']['true_initial_log_p_over_v']==-.05


def test_measured_preparation_and_failed_free_methods_inherit_exactly(parent,cfg,tmp_path,monkeypatch):
    root,path,old=parent;spec=run.initial_group_specs(cfg,'measured')[0];original=add_group(path,old,spec,arrays=True)
    out=tmp_path/'out';source=path/'prepared'/f"{spec['group']}.npz"
    result=run.prepare_fixed_roi_flow_group((cfg,out,root,spec,False))
    assert {k:v for k,v in result.items() if k!='inheritance'}==original
    assert source.read_bytes()==(out/'prepared'/source.name).read_bytes()
    selected=run.inherit_flow_training(cfg,root,out,spec,method='trained_tau_logflow1')
    assert selected['status']=='failed_training' and selected['terminal_starts']==3
    from src.inference import shared_driver_reconstruction as lib
    monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',lambda *a,**k:pytest.fail('inherited fit recomputed'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('inherited validation opened arrays'))
    for method in run.FIXED_ROI_INITIAL_FREE:
        result=run.nonlinear_cell((cfg,out,spec,method,'full',False))
        assert result['inheritance']['status']=='inherited_nonindependent'
        assert len(result['rows'])==6 and all(r['status']=='failed_training' for r in result['rows'])


@pytest.mark.parametrize('method',run.FIXED_ROI_INITIAL_TRAINED)
def test_training_tie_routes_to_reduced_solver_and_remains_train_only(cfg,tmp_path,monkeypatch,method):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.initial_group_specs(cfg,'synthetic')[0];record=metadata(cfg,spec);calls=[]
    def fit(y,p,dt,**kw):
        assert y.shape==(18,120,3) and np.all(y==2.)
        assert kw['tie_total_hb_to_volume']==method.endswith('_tied')
        assert kw['flow_prior_weight']==1. and kw['flow_prior_log_sd']==np.log(2.)
        assert kw['parameter_prior_mean'] is None and kw['parameter_prior_log_sd'] is None
        assert kw['record_trace'] is True and kw['max_evaluations']==3600
        calls.append(y.copy())
        return dict(status='failed_numerical',converged=False,parameter_value=2.,expected_trials=18,completed_trials=18)
    monkeypatch.setattr(lib,'fit_nonlinear_shared_parameter',fit)
    for folder,validation_value in [('a',3.),('b',1e9)]:
        out=tmp_path/folder;run.write_json(out/'prepared'/f"{spec['group']}.json",record)
        target=np.full((24,120,3),2.);target[record['metadata']['validation']]=validation_value
        np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=np.ones(3))
        result=run.train_gain_prior_start((cfg,out,spec,method,1,2.))
        assert result['status']=='failed_numerical' and result['tie_total_hb_to_volume']==method.endswith('_tied')
    np.testing.assert_array_equal(*calls)


@pytest.mark.parametrize('method',run.FIXED_ROI_INITIAL_METHODS)
def test_synthetic_validation_fits_all_arms_and_re_solves_tied_linear_initialization(cfg,tmp_path,monkeypatch,method):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.initial_group_specs(cfg,'synthetic')[0];record=metadata(cfg,spec)
    run.write_json(tmp_path/'prepared'/f"{spec['group']}.json",record)
    np.savez(tmp_path/'prepared'/f"{spec['group']}.npz",normalizer=np.ones(3),target=np.zeros((24,120,3)),
        feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)),truth=np.ones((24,120)),truth_states=np.ones((24,120,6)))
    if method.startswith('trained'):
        run.write_json(run.gain_training_directory(tmp_path,spec,method)/'selection.json',dict(status='completed',parameter_value=3.))
    calls=[];state=np.tile([0.,0.,1.,1.02,1.02,.99],(120,1))
    monkeypatch.setattr(run,'build_shared_driver_design',lambda *a,**k:None)
    def linear(y,design,**kw):
        assert kw['tie_total_hb_to_volume']==method.endswith('_tied')
        calls.append('linear_re_solve');return dict(driver=np.zeros(120),states=state)
    def fit(y,p,dt,**kw):
        assert p.free.tau==(3. if method.startswith('trained') else 2.)
        assert kw['tie_total_hb_to_volume']==method.endswith('_tied') and kw['flow_prior_weight']==1.
        np.testing.assert_array_equal(kw['starts'][0]['initial_state'],state[0,1:])
        assert len(kw['starts'])==2
        calls.append('nonlinear');return dict(status='failed_numerical',converged=False,starts=[])
    monkeypatch.setattr(run,'fit_shared_driver',linear);monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',fit)
    result=run.nonlinear_cell((cfg,tmp_path,spec,method,'full',False))
    assert len(calls)==12 and len(result['rows'])==6 and 'inheritance' not in result
    assert all(r['tie_total_hb_to_volume']==method.endswith('_tied') for r in result['rows'])
    description=result['trial_results'][0]['linear_start']['coordinate_fit']
    assert ('reduced_SVD' in description)==method.endswith('_tied')


def test_measured_gate_requires_all_new_synthetic_conditions(parent,cfg,monkeypatch):
    root,_,_=parent
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('measured read before synthetic terminal'))
    args=SimpleNamespace(config=CONFIG,fixed_roi_initial_fit=True,check_only=False,pilot=False,phase='measured',workers=1,
        project_root=root,run_dir=root/cfg['output_root']/'fixture')
    with pytest.raises(ValueError,match='synthetic terminal'):run.gain_prior_fit_main(args)


def test_declared_plan_counts_distinguish_inheritance(cfg,tmp_path):
    specs={kind:run.initial_group_specs(cfg,kind) for kind in ['synthetic','measured']}
    plan=[(s,m,v) for kind in specs for s in specs[kind] for m in cfg['methods'] for v in cfg['modes']]
    assert len(plan)==576
    summary=run.summarize_fixed_roi_tau(tmp_path,plan,False)
    assert summary['expected_rows']==3456 and summary['expected_training_groups']==96
    assert summary['expected_inherited_baseline_rows']==432 and summary['expected_new_validation_rows']==3024
    assert summary['expected_new_training_starts']==252 and summary['expected_inherited_training_starts']==36
    syn=run.summarize_fixed_roi_tau(tmp_path,[p for p in plan if p[0]['kind']=='synthetic'],False)
    assert syn['expected_rows']==2592 and syn['expected_new_training_starts']==216
    assert syn['expected_inherited_baseline_rows']==0 and syn['expected_inherited_training_starts']==0
