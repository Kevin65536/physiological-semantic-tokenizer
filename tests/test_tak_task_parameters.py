"""Task-parameter controls use synthetic fixtures, never local measured evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


@pytest.fixture
def configuration(tmp_path):
    path=runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_task_parameters_v1.yaml'
    cfg=yaml.safe_load(path.read_text());parent=tmp_path/cfg['parent'];parent.mkdir(parents=True)
    previous,core=runner.tak_repeatability_config(None)
    (parent/'resolved_config.json').write_text(json.dumps(previous))
    (parent/'resolved_model_config.json').write_text(json.dumps(core))
    return runner.tak_task_parameter_config(None,tmp_path)


def test_candidate_parameter_names_starts_and_complete_null_matrix(configuration,monkeypatch):
    cfg,core=configuration
    specs=runner.tak_task_parameter_specs(core,None,'synthetic')
    assert len(specs)==108 and len({s['key'] for s in specs})==108
    arm=core['arms'][-1];seen={}
    import src.inference.shared_driver_reconstruction as model
    def fit(*args,**kwargs):seen.update(kwargs);return {}
    monkeypatch.setattr(model,'fit_nonlinear_shared_parameters',fit)
    y=np.zeros((1,10,3))
    runner.tak_fit_group(core,arm,y,np.eye(30)[None],np.ones((1,3)),np.ones(y.shape,bool))
    assert seen['parameter_names']==['tau','neurovascular_gain']
    np.testing.assert_array_equal([s['parameter_values'] for s in seen['starts']],cfg['model']['tau_beta_starts'])
    np.testing.assert_array_equal(seen['parameter_prior_log_mean'],np.log([2,1]))
    assert seen['data_weight']==pytest.approx(.4)


def test_synthetic_changes_only_declared_physiology_with_matched_noise(configuration):
    _,core=configuration
    case=dict(subject=1,repeat=0,task='B',scenario='unchanged_parameters')
    y,op,sd,truth=runner.tak_task_parameter_synthetic(core,case)
    for scenario,param in [('tau_shift','tau'),('beta_shift','beta')]:
        z,other,scale,t=runner.tak_task_parameter_synthetic(core,dict(case,scenario=scenario))
        np.testing.assert_array_equal(t['driver'],truth['driver'])
        np.testing.assert_array_equal(t['initial_state'],truth['initial_state'])
        np.testing.assert_allclose(z-t['clean'],y-truth['clean'],atol=2e-17)
        np.testing.assert_array_equal(other,op);np.testing.assert_array_equal(scale,sd)
        assert t[param]/truth[param]==pytest.approx(1.5)
        assert t['beta' if param=='tau' else 'tau']==truth['beta' if param=='tau' else 'tau']
    _,_,_,a=runner.tak_task_parameter_synthetic(core,dict(case,task='A'))
    assert (a['tau'],a['beta'])==(truth['tau'],truth['beta'])
    assert not np.array_equal(a['driver'],truth['driver'])


def test_task_preparation_rejects_incomplete_synthetic_before_signal_access(tmp_path,monkeypatch):
    plan={'fixture':True};monkeypatch.setattr(runner,'tak_task_parameter_plan',lambda *args:plan)
    (tmp_path/'synthetic_summary.json').write_text(json.dumps(dict(status='incomplete',terminal_rows=0)))
    with pytest.raises(ValueError,match='Complete synthetic'):
        runner.tak_task_parameter_prepare({}, {},tmp_path,tmp_path,plan,1)


def test_candidate_statistics_accept_beta_without_relabelling_kappa(configuration):
    import pandas as pd
    rows=[]
    for s in range(3):
        for i in range(3):
            rows.append(dict(dataset='fixture',analysis_tier='primary',subject=str(s),native_group=f'{s}-{i}',
                qualified=True,fixed_qualified=True,tau=2+.1*s+.01*i,beta=1+.2*s+.02*i,
                full_hb_nmse=.1,fixed_full_hb_nmse=.2))
    subjects,stats,_=runner._tak_repeatability_statistics(pd.DataFrame(rows),parameters=('tau','beta'),bootstrap_repeats=20)
    assert set(subjects.parameter)=={'tau','beta'} and stats.included_subjects.eq(3).all()


def test_task_readout_leaves_support_out_and_keeps_missing_in_denominator():
    x=np.zeros((6,3,2,2))
    for s in range(6):
        x[s,:,:,0]=np.array([s*10.,s*10.+2.])[None]
        x[s,:,:,1]=np.arange(3)[:,None]*.01
    result=runner._tak_task_discrimination(x,permutations=199)
    assert result['correct']==36 and result['abstentions']==0 and result['accuracy']==1
    assert result['permutation_p']<.05
    # Missing task in one subject causes abstentions, never a reduced denominator.
    x[0,:,1]=np.nan
    result=runner._tak_task_discrimination(x,permutations=199)
    assert result['planned_tests']==36 and result['correct']==30 and result['abstentions']==6
    assert result['accuracy']==pytest.approx(30/36)
    # Swapping the held-out support swaps its decisions, not its training centroids.
    x[1,0]=x[1,0,::-1]
    prediction=runner._tak_task_predictions(x)
    np.testing.assert_array_equal(prediction[1,0],[1,0])


def test_equal_parameters_cannot_classify_by_first_label_tie_breaking():
    result=runner._tak_task_discrimination(np.zeros((6,3,3,2)),permutations=20)
    assert result['accuracy']==0 and result['abstentions']==54 and result['permutation_p']==1


def test_analysis_checks_array_scores_identity_and_common_three_arm_denominator(configuration,tmp_path):
    cfg,core=configuration;run=tmp_path/'fixture_run';run.mkdir();(run/'tasks').mkdir()
    cfg=deepcopy(cfg);cfg['endpoints']['classifier']['null_repeats']=20
    plan=dict(records=[],groups=[],annotations={},parent_window_ids=[],excluded_new_windows=0,excluded=[])
    first=None
    for dataset,tasks in cfg['endpoints']['task_sets'].items():
        for subject in ['s0','s1']:
            for ti,task in enumerate(tasks):
                for repeat in range(3):
                    for subset in range(2):
                        key=f'{dataset}_{subject}_{task}_{repeat}_{subset}'
                        ref=dict(id=key,dataset=dataset,subject=subject,record=task,native_group=f'{dataset}_{subject}_{task}_{repeat}',
                            subset_index=subset,analysis_tier='primary' if ti==0 else 'task_extension',
                            eeg_start_s=100*subset,hb_start_s=100*subset,anchor={'hb_channel':'fixture'})
                        group=dict(id=key,refs=[ref]);plan['records'].append(ref);plan['groups'].append(group)
                        plan['annotations'][key]=dict(task=task,task_repeat=repeat,scope='parent' if ti==0 else 'new_task')
                        if ti==0:plan['parent_window_ids'].append(key)
                        for arm in core['arms']:
                            path=run/'tasks'/f'measured__{key}__90__joint_full__{arm["name"]}.json'
                            good=not (first is None and arm['name']=='tak_tau_beta')
                            if not good:first=path
                            row=dict(id=key,arm=arm['name'],qualified=good,fit_status='passed' if good else 'not_converged',
                                kappa=.64+.1*ti,tau=2+.2*ti,beta=1+.1*ti,full_hb_nmse=0.,eeg_nmse=0.,integration_max_training_sd=0.)
                            runner.write_json(path,dict(spec=dict(group=group,arm=arm),rows=[row],fit={}))
                            if good:np.savez_compressed(path.with_suffix('.npz'),target=np.zeros((1,900,3)),prediction=np.zeros((1,900,3)),sd=np.ones((1,3)))
    for spec in runner.tak_task_parameter_specs(core,plan,'synthetic'):
        row=dict(qualified=True,tau=2.,beta=1.,truth_tau=2.,truth_beta=1.,full_hb_nmse=0.,integration_max_training_sd=0.)
        runner.write_json(run/'tasks'/(spec['key']+'.json'),dict(rows=[row]))
    for name,value in [('resolved_config',cfg),('resolved_model_config',core),('plan',plan),
                       ('measured_summary',dict(status='completed',terminal_rows=len(plan['records'])*3))]:
        runner.write_json(run/(name+'.json'),value)
    result=runner.tak_task_parameter_analyze(run,run/'analysis')
    assert result['checked_array_rows']==len(plan['records'])*3-1
    assert len(result['failures'])==1 and len(result['classification'])==4
    trial=next(r for r in result['reconstruction'] if r['dataset']=='eeg_fnirs_single_trial')
    assert trial['planned_windows']==12 and trial['paired_windows']==11
    bad=json.loads(first.read_text());bad['rows'][0]['id']='wrong_identity';runner.write_json(first,bad)
    with pytest.raises(ValueError,match='identity'):
        runner.tak_task_parameter_analyze(run,run/'bad_analysis')
