"""Synthetic/temporary fixtures only; no retained measured evidence required."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_ssm_strengthening as runner


@pytest.fixture
def configs():
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    base = yaml.safe_load((runner.CODE_ROOT/cfg['source_config']).read_text())
    return cfg,base


def test_contract_and_masks_fail_closed_before_array_access(configs,tmp_path):
    cfg,_ = configs
    bad = deepcopy(cfg); bad['source_run']='comparative_methods/protected'
    path = tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError,match='boundary'):
        runner.read_config(path)
    bad = deepcopy(cfg);bad['synthetic']['seed_streams']['evaluation']=bad['synthetic']['seed_streams']['train']
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError,match='seed streams'):
        runner.read_config(path)
    mask = runner.visible_mask('Hb_hidden')
    assert mask[:,0].all() and not mask[:,1:].any()
    assert (~runner.visible_mask('center_Hb')).sum()==32
    with pytest.raises(ValueError):
        runner.visible_mask('future_native')
    ref=dict(array_path=str(tmp_path/'protected.npz'),dataset='eeg_fnirs_single_trial',subject='subject_01',record='r')
    with pytest.raises(ValueError,match='before array'):
        runner.validate_public_ref(ref,tmp_path,cfg['measured']['datasets'])


def test_synthetic_response_and_observation_counterfactuals_are_paired(configs):
    cfg,base = configs
    args = (cfg,base,'H0','evaluation',1,'mixed','nonrest')
    control = runner.synthetic_case(*args,'shared_neural_only')
    common = runner.synthetic_case(*args,'common_correlated')
    np.testing.assert_array_equal(control['driver'],common['driver'])
    np.testing.assert_array_equal(control['physical_truth'],common['physical_truth'])
    np.testing.assert_allclose(common['target']-control['target'],common['observation_truth'],atol=1e-16)
    changed = runner.synthetic_case(cfg,base,'Htau_kappa','evaluation',1,'mixed','nonrest','common_correlated')
    np.testing.assert_array_equal(changed['driver'],common['driver'])
    np.testing.assert_array_equal(changed['observation_truth'],common['observation_truth'])
    assert np.linalg.norm(changed['physical_truth']-common['physical_truth'])>1e-3
    training = runner.synthetic_case(cfg,base,'H0','train',1,'mixed','nonrest','shared_neural_only')
    assert not np.array_equal(training['driver'],control['driver'])
    assert training['subject'] != control['subject']
    dirty = runner.synthetic_case(*args,'donor_contamination')
    np.testing.assert_array_equal(dirty['target'],control['target'])
    assert np.linalg.norm(dirty['donors']-control['donors'])>1e-3


def test_candidate_choice_excludes_entire_subject_and_preserves_failure_denominator():
    rows = [dict(subject=s,candidate=i,component=False,score_nrmse=float(i),converged=True)
            for s in ('A','B','C') for i in range(9)]
    for row in rows:
        if row['subject']=='A':row['score_nrmse']=0 if row['candidate']==8 else 100.
    result = runner.response_choice(rows,exclude_subjects=['A'])
    assert result['selected']==0 and result['selection_subjects']==['B','C']
    changed = deepcopy(rows)
    for row in changed:
        if row['subject']=='A':row['score_nrmse']=1e15
    assert runner.response_choice(changed,exclude_subjects=['A'])==result
    for row in changed:
        if row['candidate']==0 and row['subject']=='B':row.update(converged=False,score_nrmse=0.)
    assert runner.response_choice(changed,exclude_subjects=['A'])['selected']==1


def test_independent_donors_reject_target_alias_wrong_clock_and_duplicate_pair():
    ref=dict(id='t',key='record',window=0,site='A',hb_channel='H1',dataset='public',subject='s',eeg_start_s=10.,hb_start_s=11.)
    good=dict(ref,id='d',site='B',hb_channel='H2')
    alias=dict(ref,id='alias',site='C')
    duplicate=dict(good,id='duplicate',site='D')
    badtime=dict(good,id='later',site='E',hb_channel='H3',hb_start_s=12.)
    assert [d['id'] for d in runner.synchronized_donors(ref,[ref,good,alias,duplicate,badtime])]==['d']


def test_prepared_reader_never_regenerates_missing_inputs(tmp_path):
    from src.data.ssm_prepared import prepared_target,read_prepared_arrays
    path=tmp_path/'absent.npz'
    with pytest.raises(FileNotFoundError):
        read_prepared_arrays(str(path))
    eeg=np.arange(120*30,dtype=float).reshape(1,120,30);hb=np.ones((1,120,2))
    np.savez(path,eeg_features=eeg,hb=hb)
    coordinate=dict(pc=np.eye(30)[0],eeg_factor=.1,hb_factor=2.)
    y=prepared_target(dict(array_path=str(path),array_index=0),coordinate)
    np.testing.assert_array_equal(y[:,0],eeg[0,:,0]*.1)
    np.testing.assert_array_equal(y[:,1:],2.)
    assert not read_prepared_arrays(str(path))[0].flags.writeable


def test_frozen_spatial_pipeline_has_no_hidden_target_feedback(configs,tmp_path):
    cfg,base=configs
    cal=runner.synthetic_calibration(cfg,base)
    case=runner.attach_scales(runner.synthetic_case(cfg,base,'Htau','pilot',5,'mixed','nonrest','common_colored'),cal)
    rng=np.random.default_rng(78)
    x=rng.normal(size=(120,4));y=.01*x[:,:2]
    models={}
    signature='|'.join(case['donor_signature'])
    for regime in ('H0','Hselected'):
        maps={str(rank):{signature:runner.fit_reduced_rank_readout(x,y,rank=rank,weight=.1)} for rank in (1,2)}
        models[regime]=dict(selected_rank=1,by_rank=maps)
    frozen=dict(response=dict(selected=2),calibration=cal,training_keys=[],sensitivity_threshold=1.,
        spatial=dict(models=models,baselines={},template=np.zeros((120,3))))
    reference=runner.evaluate_case(cfg,base,case,frozen,tmp_path,['Hb_hidden'])
    modified=deepcopy(case);modified['target'][:,1:]=1e50
    changed=runner.evaluate_case(cfg,base,modified,frozen,tmp_path,['Hb_hidden'])
    for key,value in reference['arrays'].items():
        np.testing.assert_array_equal(value,changed['arrays'][key])
    assert np.any(reference['arrays']['Hb_hidden__F01__observation_component']!=0)
    unavailable=deepcopy(case)
    unavailable.update(donors=[],donor_sds=[],donor_ratios=[],donor_signature=[],raw_donors=np.empty((120,0)))
    missing=runner.evaluate_case(cfg,base,unavailable,frozen,tmp_path,['Hb_hidden'])
    failed=[r for r in missing['rows'] if r['arm']=='F01' and r['pairing']=='real']
    assert len(failed)==1 and not failed[0]['converged']
    assert failed[0]['status']=='missing_donor_or_training_signature'


def test_raw_spatial_baseline_remains_available_when_all_SSM_fits_fail(configs,tmp_path,monkeypatch):
    cfg,base=configs
    cal=runner.synthetic_calibration(cfg,base)
    case=runner.attach_scales(runner.synthetic_case(cfg,base,'H0','pilot',4,'slow','rest','common_colored'),cal)
    def failed_model(*args,**kwargs):
        z=np.zeros((120,3))
        return dict(converged=False,status='fixture_fit_failed',prediction=z,physical_prediction=z,
                    observation_component=z,driver=np.zeros(120))
    monkeypatch.setattr(runner,'fit',failed_model)
    monkeypatch.setattr(runner,'donor_features',lambda *args:(np.zeros((120,4)),False,[dict(converged=False)]))
    signature='|'.join(case['donor_signature'])
    ridge=runner.fit_standardized_ridge(case['raw_donors'],case['target'][:,1:],.1)
    frozen=dict(response=dict(selected=0),calibration=cal,training_keys=[],sensitivity_threshold=1.,
        spatial=dict(models={r:dict(selected_rank=1,by_rank={'1':{},'2':{}}) for r in ('H0','Hselected')},
                     baselines={signature:dict(ridge=ridge)},template=np.zeros((120,3))))
    result=runner.evaluate_case(cfg,base,case,frozen,tmp_path,['Hb_hidden'])
    rows={r['pairing']:r for r in result['rows'] if r['arm']=='spatial_ridge'}
    assert all(rows[p]['converged'] for p in ('real','real_shift_support','nonwrapping_shift_12s'))
    assert not rows['wrong_subject']['converged']
    direct=runner.predict_standardized_ridge(ridge,case['raw_donors'])
    expected=runner.rmse((direct-case['target'][:,1:])/cal['sd'][1:])
    assert rows['real']['score_nrmse']==pytest.approx(expected)
    shifted=runner.spatial_ridge_result(direct,'nonwrapping_shift_12s')['prediction']
    np.testing.assert_array_equal(shifted[:72,1:],direct[48:])
    np.testing.assert_array_equal(shifted[72:],np.zeros((48,3)))


def test_training_crossfit_selection_and_summary_use_full_denominators(configs,tmp_path,monkeypatch):
    cfg,base=configs
    cfg=deepcopy(cfg);cfg['synthetic']['train_repeats']=3;cfg['synthetic']['selection_repeats']=3
    cfg['synthetic']['training_scenarios']=['shared_neural_only']
    cal=runner.serial(runner.synthetic_calibration(cfg,base))

    def cheap_fit(base,target,sd,cal,candidate=0,*,mode='full',component=False,mean=None,**kwargs):
        # Explicit visible-EEG-only surrogate keeps this orchestration test
        # independent of expensive numerical evidence and hidden target values.
        physical=np.zeros_like(target)
        if runner.visible_mask(mode)[:,0].all():
            physical[:,0]=target[:,0]
            physical[:,1]=.1*target[:,0]/(1+candidate)
            physical[:,2]=-.02*target[:,0]/(1+candidate)
        c=np.zeros_like(target) if mean is None else mean
        return dict(converged=True,status='fixture',prediction=physical+c,physical_prediction=physical,
                    observation_component=c,driver=physical[:,0],objective=0.,evaluations=1)
    monkeypatch.setattr(runner,'fit',cheap_fit)
    plan=runner.synthetic_plans(cfg)
    key='synthetic__H0'
    specs=[s for s in plan['training'] if s['group']==key]
    for spec in specs:
        result=runner.training_worker((cfg,base,str(tmp_path),spec,cal))
        runner.write_json(tmp_path/'training'/f'{spec["key"]}.json',result)
    group=plan['groups'][key]
    frozen=runner.calibration_worker((cfg,str(tmp_path),key,group['training'],group['selection'],cal))
    runner.write_json(tmp_path/'calibration'/f'{key}.json',frozen)
    assert not set(frozen['training_subjects'])&set(frozen['selection_subjects'])
    for fold in frozen['spatial']['crossfit']:
        assert not set(fold['excluded_subjects'])&set(fold['selection_subjects'])
    spec=next(s for s in plan['evaluation'] if s['group']==key)
    result=runner.evaluation_worker((cfg,base,str(tmp_path),spec))
    runner.write_json(tmp_path/'synthetic'/f'{spec["key"]}.json',result)
    assert len(result['rows'])==len(runner.expected_case_rows('synthetic',cfg))*len(cfg['synthetic']['scenarios'])
    frame,failures=runner.collect_metrics(cfg,tmp_path,'synthetic',[spec])
    assert not failures
    paired=runner.paired_summary(cfg,frame)
    assert not paired.empty
    direct_pairs=paired[paired.baseline.isin(['spatial_ridge','training_template'])]
    assert not direct_pairs.empty
    assert set(direct_pairs.metric)<= {'score_nrmse','Hb_nrmse','HbO_nrmse','HbR_nrmse'}
    assert not runner.null_summary(cfg,frame).empty
    assert not runner.risk_table(cfg,frame).empty
    attribution=runner.attribution_table(cfg,tmp_path,[spec])
    assert not attribution.empty
    projections=attribution[[f'observed_to_{name}_projection' for name in ('physical','component','residual')]].dropna()
    np.testing.assert_allclose(projections.sum(axis=1),1.,atol=1e-12)
    attribution.to_csv(tmp_path/'mechanism_attribution.csv',index=False)
    runner.calibration_tables(cfg,base,tmp_path,[key])
    decision=runner.make_decision(cfg,tmp_path,runner.paired_summary(cfg,frame),
                                  runner.null_summary(cfg,frame),runner.profile_summary([]))
    assert decision['S5']['fresh_confirmation']=='not_launched'
    failed_spec=dict(spec,key=spec['key']+'_unavailable')
    with_failure,failures=runner.collect_metrics(cfg,tmp_path,'synthetic',[spec,failed_spec])
    assert len(failures)==1 and len(with_failure)==2*len(frame)
    assert int((~with_failure.converged).sum())>=len(frame)

    # Correct a retained baseline failure into a new version without invoking
    # any fitter or changing other methods' scientific evidence.
    for row in result['rows']:
        if row['arm']=='spatial_ridge':
            row.update(converged=False,status='old_inherited_SSM_failure',score_nrmse=None)
    runner.write_json(tmp_path/'synthetic'/f'{spec["key"]}.json',result)
    source_plan=dict(plan,evaluation=[spec],groups={key:plan['groups'][key]},profiles=[])
    runner.write_json(tmp_path/'synthetic_plan.json',source_plan)
    runner.write_json(tmp_path/'measured_plan.json',dict(groups={},profile_ids=[]))
    for name in ('baseline_profile_summary','software_probe','historical_source_audit','input_identity','verification'):
        runner.write_json(tmp_path/f'{name}.json',dict(status='passed'))
    runner.write_json(tmp_path/'run_manifest.json',dict(execution='completed'))
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    corrected=tmp_path.with_name(tmp_path.name+'_corrected');corrected.mkdir()
    def forbidden_fit(*args,**kwargs):
        raise AssertionError('baseline correction must not refit SSMs')
    monkeypatch.setattr(runner,'fit',forbidden_fit)
    monkeypatch.setattr(runner,'summarize',lambda *args:None)
    monkeypatch.setattr(runner,'verify',lambda *args:None)
    runner.correct_spatial_ridge(cfg,base,tmp_path,corrected)
    revised=runner.read_json(corrected/'synthetic'/f'{spec["key"]}.json')
    assert [r for r in result['rows'] if r['arm']!='spatial_ridge']==[r for r in revised['rows'] if r['arm']!='spatial_ridge']
    assert any(r['converged'] for r in revised['rows'] if r['arm']=='spatial_ridge')
    assert revised['array_path']==result['array_path']
    assert runner.read_json(corrected/'correction.json')['repeated_SSM_fits']==0
