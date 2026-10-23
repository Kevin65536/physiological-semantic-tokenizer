"""Feasibility dispatch and retained-public boundaries use temporary evidence."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_source_observability as runner


@pytest.fixture
def cfg():
    return runner.feasibility_config(runner.FEASIBILITY_CONFIG)


@pytest.mark.parametrize('field,value', [
    ('source_run', 'experiments/runs/protected'),
    ('protected_data', 'allowed'),
    ('raw_or_cache_reads', 'allowed'),
])
def test_config_rejects_expanded_data_boundary(cfg, tmp_path, field, value):
    changed = deepcopy(cfg)
    changed['sources'][field] = value
    path = tmp_path/'invalid.yaml'
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        runner.feasibility_config(path)


def test_check_is_array_and_metadata_free(cfg, monkeypatch):
    monkeypatch.setattr(runner, 'feasibility_source', lambda *_: pytest.fail('metadata read'))
    monkeypatch.setattr(runner.np, 'load', lambda *_a, **_kw: pytest.fail('array read'))
    result = runner.feasibility_synthetic(cfg)
    assert result['status'] == 'passed'
    assert result['measured_arrays_read'] == 0
    assert result['representable_EEG_nrmse'] < 1e-10
    assert result['incompatible_common_Hb_nrmse'] > .001


def test_metrics_preserve_training_scale_and_report_shape_errors():
    target = np.tile(np.linspace(-1, 1, 120)[:, None], (1, 68))
    prediction = target + 1.
    result = runner.feasibility_metrics(prediction, target, np.full(68, 2.))
    for modality in ('EEG', 'HbO', 'HbR'):
        assert result[modality]['nrmse'] == pytest.approx(.5)
        assert result[modality]['centered_nrmse'] < 1e-15
        assert result[modality]['mean_feature_correlation'] == pytest.approx(1.)
    with pytest.raises(ValueError):
        runner.feasibility_metrics(prediction[:, :3], target[:, :3], np.ones(3))


def test_prepared_metadata_mismatch_precedes_array_read(tmp_path, cfg, monkeypatch):
    row = dict(id='ordinary|subject|record', split='evaluation')
    source = dict(path=tmp_path, plan=dict(rows=[row]))
    name = hashlib.sha256(row['id'].encode()).hexdigest()[:24]
    path = tmp_path/'prepared'/(name+'.json')
    path.parent.mkdir()
    path.write_text(json.dumps(dict(status='completed', row=dict(row, split='train'))))
    monkeypatch.setattr(runner.np, 'load', lambda *_a, **_kw: pytest.fail('array read before identity'))
    with pytest.raises(ValueError, match='before array read'):
        runner.feasibility_load(source, row)


def test_plan_keeps_all_windows_and_both_arms(cfg):
    rows = [dict(id=f'evaluation{i}', split='evaluation', subject=f's{i//2}', record=f'r{i%2}') for i in range(6)]
    rows += [dict(id='train', split='train', subject='train', record='r')]
    plan = runner.feasibility_plan(cfg, dict(plan=dict(rows=rows)))
    assert plan['planned'] == 12
    assert plan['evaluation_subjects'] == 3
    assert len({task['id'] for task in plan['tasks']}) == 12
    assert all(task['row']['split'] == 'evaluation' for task in plan['tasks'])


def test_aggregation_weights_records_and_subjects_before_square_root():
    records = [dict(row=dict(subject='A', record='first'), value=1.),
               dict(row=dict(subject='A', record='first'), value=1.),
               dict(row=dict(subject='A', record='second'), value=9.),
               dict(row=dict(subject='B', record='only'), value=3.)]
    result = runner.feasibility_aggregate(records, lambda row: row['value'], repeats=20)
    assert result['subject_nmse'] == dict(A=5., B=3.)
    assert result['nrmse'] == pytest.approx(2.)
    assert result['windows'] == 4


def test_failed_task_is_retained_and_resumed(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'feasibility_source', lambda *_: (_ for _ in ()).throw(ValueError('source rejection')))
    task = dict(id='test', kind='bounds', arm='K1_true_H4_E12', row=dict(id='r', subject='s', record='r'))
    result = runner.feasibility_execute((cfg, str(tmp_path), 'bounds', task))
    assert result['status'] == 'failed_exception'
    assert 'source rejection' in result['error']
    monkeypatch.setattr(runner, 'feasibility_source', lambda *_: pytest.fail('resume repeated source read'))
    assert runner.feasibility_execute((cfg, str(tmp_path), 'bounds', task)) == result


def test_small_budget_synthetic_fit_integrates_both_library_apis(cfg, monkeypatch):
    """Exercise actual operators, dataclass metadata, and fixed/free fit calls."""
    changed=deepcopy(cfg)
    changed['solver'].update(max_nfev=2,vascular_max_nfev=2)
    parent=dict(fixed=dict(tau=2.,kappa=.64,gamma=.32,alpha=.32,E0=.32,
        P0=1.,Q0=.35,neurovascular_gain=1.,eeg_loading=1.),solver=dict(numerical_backend='numba'))
    loading=np.linspace(-1.,1.,60)[:,None]
    source=dict(parent=parent,coordinate=dict(sd=np.full(68,.025)),calibrations={
        'K1_true_H4_E12':dict(readout=dict(eeg_loadings=loading),design=dict(hb_loadings=np.ones((4,1))))})
    monkeypatch.setattr(runner.np,'load',lambda *_a,**_kw:pytest.fail('synthetic read a retained array'))
    task=runner.feasibility_synthetic_plan(changed,pilot=True)[0]
    result,arrays=runner.feasibility_synthetic_task(changed,source,task)
    assert result['success']
    assert result['measured_arrays_read']==0
    assert set(result['frontier'])=={'fixed','free_tau_kappa'}
    assert result['vascular'] is not None
    assert arrays['target'].shape==(120,68)
    json.dumps(result,allow_nan=False)


def test_profile_manifest_records_not_entered_when_no_joint_witness(cfg,tmp_path):
    row=dict(id='public-evaluation',split='evaluation',subject='s',record='r')
    source=dict(plan=dict(rows=[row]))
    for arm in runner.FEASIBILITY_ARMS:
        identity=runner.token(dict(kind='frontier',record_id=row['id'],arm=arm))
        runner.write_json(tmp_path/'frontier'/(identity+'.json'),dict(joint_witness_found=False))
    assert runner.feasibility_fit_plan(cfg,source,tmp_path,'profiles')==[]
    assert runner.read_json(tmp_path/'profiles_manifest.json')['execution']=='not_entered'


def test_summary_retains_failed_stage_denominators(cfg,tmp_path):
    row=dict(id='public-evaluation',split='evaluation',subject='s',record='r')
    plan=runner.feasibility_plan(cfg,dict(plan=dict(rows=[row])))
    runner.write_json(tmp_path/'manifest.json',plan)
    task=dict(id='frontier-failure',kind='frontier',arm=runner.FEASIBILITY_ARMS[0],row=row)
    runner.write_json(tmp_path/'frontier_manifest.json',dict(tasks=[task],planned=1))
    runner.write_json(tmp_path/'frontier'/(task['id']+'.json'),dict(task,status='failed_exception',success=False,error='retained'))
    result=runner.feasibility_summary(cfg,tmp_path)
    assert result['execution']=='incomplete'
    assert result['profiles']['execution']=='not_entered'
    group=result['frontier'][runner.FEASIBILITY_ARMS[0]+'|fixed|epsilon=0.5']
    assert group['planned']==1
    assert group['task_exceptions']==1
    assert group['legal_EEG_feasible_witnesses']==0
    assert group['metrics']['HbO']['nrmse'] is None
    json.dumps(result,allow_nan=False)


def test_state_metadata_keeps_engineering_extrema_and_physical_initials():
    states=np.ones((3,2,6));states[:,0,2]=5.
    initials=np.array([[0.,np.log(5.),0.,0.,0.],[0.,0.,0.,0.,0.]])
    result=runner.feasibility_fit_metadata(dict(states=states,initial_coordinates=initials,tau=.6,kappa=.64))
    diagnostics=result['state_diagnostics']
    assert diagnostics['cores'][0]['flow']['max']==5.
    assert diagnostics['flow_outside_engineering_reference']
    assert diagnostics['initial_outside_engineering_reference']
    assert diagnostics['physical_initials'][0][1]==pytest.approx(5.)
    assert diagnostics['parameter_bound_hits']['tau']


def test_original_stage_dispatch_remains_closed_to_new_stages():
    with pytest.raises(SystemExit):
        runner.main(['--stage','bounds'])


def test_frontier_comparison_uses_same_record_subset(cfg,tmp_path):
    rows=[dict(id=f'row{i}',split='evaluation',subject=f's{i}',record='r') for i in range(2)]
    runner.write_json(tmp_path/'manifest.json',runner.feasibility_plan(cfg,dict(plan=dict(rows=rows))))
    tasks=[]
    for index,row in enumerate(rows):
        task=dict(id=f'frontier{index}',row=row,arm=runner.FEASIBILITY_ARMS[0])
        tasks.append(task)
        nodes=[]
        for branch,nmse in ([('fixed',4.),('free_tau_kappa',1.)] if index==0 else [('free_tau_kappa',100.)]):
            fit=dict(witness_feasible=True,physical_valid=True,integration_error=0.,starts=[],
                scores={m:dict(nmse=nmse,nrmse=np.sqrt(nmse)) for m in ('EEG','Hb','HbO','HbR')})
            nodes.append(dict(branch=branch,epsilon=.5,status='completed',witness_feasible=True,fit=fit))
        runner.write_json(tmp_path/'frontier'/(task['id']+'.json'),dict(task,nodes=nodes,success=True,status='completed'))
    runner.write_json(tmp_path/'frontier_manifest.json',dict(tasks=tasks,planned=2))
    result=runner.feasibility_summary(cfg,tmp_path)
    paired=result['paired_frontier'][runner.FEASIBILITY_ARMS[0]+'|epsilon=0.5']
    assert paired['paired_windows']==1
    assert paired['missing_or_not_jointly_EEG_feasible']==1
    assert paired['metrics']['Hb']['fixed']['nrmse']==pytest.approx(2.)
    assert paired['metrics']['Hb']['free']['nrmse']==pytest.approx(1.)
    assert paired['metrics']['Hb']['paired_nmse_fixed_minus_free']['mean']==pytest.approx(3.)
    assert paired['nesting_violations']==0
