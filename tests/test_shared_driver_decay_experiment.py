"""Fixture-only runner contracts; no retained or protected evidence is needed."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as run
from src.inference import shared_driver_reconstruction as fitting
from src.inference.shared_driver_rk4 import decay_forward


@pytest.fixture
def cfg():
    return run.decay_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_decay_v1.yaml')


def test_config_rejects_expansion_and_changed_priors_before_array_reads(cfg,tmp_path,monkeypatch):
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('unexpected array read'))
    for section,key,value in [('data','raw_reads',True),('data','protected_data','allowed'),
                              ('forcing','prior_sd',.1),('solver','penalty',0.),
                              ('tensor','lengths_s',[15,30,90]),('synthetic','truth_in_fitting',True)]:
        changed = deepcopy(cfg); changed[section][key] = value
        path = tmp_path/'changed.yaml'; path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError,match='boundary'):
            run.decay_config(path)
    path = run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_decay_v1.yaml'
    run.decay_main(SimpleNamespace(config=path,check_only=True))


def test_fixed_matrix_counts_and_unique_identity(cfg):
    plan = dict(records=[dict(id=f'fixture{i}',dataset='fixture',subject=f's{i//3}') for i in range(48)])
    for stage,expected in [('pilot',4),('synthetic',144),('measured',288)]:
        specs = run.decay_specs(cfg,plan,stage)
        assert len(specs) == expected
        assert len({spec['key'] for spec in specs}) == expected
    assert len(cfg['decay']['arms']) == 6
    assert [arm['multiplier'] for arm in cfg['decay']['arms']] == [1.,.1,0.,1.,.1,0.]
    assert cfg['forcing']['modes'] == {30:7,90:19}


@pytest.mark.parametrize('scenario',['matched','original_transient','slow_transient','sustained_input'])
def test_synthetic_crops_are_one_trajectory_and_no_truth_enters_starts(cfg,scenario):
    full,full_operator,sd,truth = run.decay_synthetic(cfg,0,scenario,90)
    crop,crop_operator,crop_sd,crop_truth = run.decay_synthetic(cfg,0,scenario,30)
    np.testing.assert_array_equal(sd,crop_sd)
    for name in ('driver','states','delta','forcing','noise'):
        np.testing.assert_array_equal(crop_truth[name],truth[name][120:240])
    np.testing.assert_allclose((full_operator@truth['canonical_prediction'].ravel()).reshape(360,3),truth['prediction'],atol=1e-15)
    np.testing.assert_allclose((crop_operator@crop_truth['canonical_prediction'].ravel()).reshape(120,3)[:-1],crop_truth['prediction'][:-1],atol=1e-15)
    # The same reference subtraction applies to the noiseless and noisy views.
    reference = np.eye(120); reference[:,:20] -= 1/20
    np.testing.assert_allclose(crop,reference@full[120:240],atol=2e-15)
    if scenario == 'matched':
        assert np.max(abs(truth['delta'])) < 1e-14
    elif scenario in ('original_transient','slow_transient'):
        assert truth['delta'][0] == pytest.approx(.08)
        assert np.max(abs(truth['forcing'])) == 0
    else:
        assert np.std(truth['forcing']) > .008


def test_synthetic_scenarios_share_driver_noise_and_sign_variation(cfg):
    outputs = [run.decay_synthetic(cfg,2,scenario,90)[3] for scenario in cfg['synthetic']['scenarios']]
    for other in outputs[1:]:
        np.testing.assert_array_equal(other['driver'],outputs[0]['driver'])
        np.testing.assert_array_equal(other['noise'],outputs[0]['noise'])
    positive = run.decay_synthetic(cfg,0,'sustained_input',90)[3]['forcing']
    negative = run.decay_synthetic(cfg,1,'sustained_input',90)[3]['forcing']
    np.testing.assert_allclose(negative,-1.03*positive,atol=1e-16)


@pytest.mark.parametrize('length',[30,90])
def test_metrics_use_hazard_retention_and_demeaned_matched_center(cfg,length):
    n = int(length/.25); t = np.arange(n)*.25
    target = np.column_stack((np.sin(t),np.cos(t),np.sin(.3*t)))
    prediction = target.copy(); prediction[:,1:] += np.array([2.,-3.])
    states = np.column_stack((np.zeros((n,3)),np.ones((n,3))))
    states[:,4] += 1e-14*np.sin(t)  # cancellation-prone delta never defines retention
    hazard = t/2
    forward = dict(retention_integral=hazard,decay_retention=np.exp(-hazard))
    metrics = run.decay_metrics(target,prediction,np.ones(3),'joint_full',states,np.zeros(n),forward,cfg)
    assert metrics['full_hb_nmse'] == pytest.approx(6.5)
    assert metrics['center_demeaned_hb_nmse'] < 1e-25
    assert metrics['late_demeaned_hb_nmse'] < 1e-25
    assert metrics['retention_e_fold_s'] == 2
    assert metrics['retention_95_loss_s'] == 6
    forward['decay_retention'] = np.ones(n)
    no_decay = run.decay_metrics(target,prediction,np.ones(3),'joint_full',states,np.zeros(n),forward,cfg)
    assert no_decay['retention_e_fold_s'] is None
    assert no_decay['retention_95_loss_s'] is None


def test_fixed_upstream_replay_preserves_hbr_and_constant_raw_delta_reference(cfg):
    n = 120; p = run.volume_drive_parameters(cfg); driver = .015*np.sin(np.arange(n)*.1)
    initial = np.array([0.,1.,1.,1.08,1.])
    reference = np.eye(n); reference[:,:20] -= 1/20
    operator = np.kron(reference,np.eye(3))
    forward = decay_forward(driver,initial,p,.25,substeps=4)
    original = dict(driver=driver,initial_state=initial,states=forward['states'],
        prediction=(operator@forward['canonical_prediction'].ravel()).reshape(n,3))
    arrays,audits = run.decay_fixed_upstream_replays(cfg,original,operator,np.ones(3))
    assert len(audits) == 3
    for audit in audits:
        assert audit['upstream_max_abs_difference'] < 1e-14
        assert audit['hbr_max_abs_difference'] < 1e-14
    np.testing.assert_allclose(arrays['replay_no_decay_raw_delta_contribution'][:,1],.08,atol=2e-15)
    np.testing.assert_allclose(arrays['replay_no_decay_referenced_delta_contribution'],0,atol=2e-15)
    assert abs(arrays['replay_original_referenced_delta_contribution'][-1,1]) > .01


def test_infeasible_counterfactual_is_isolated_from_valid_replays(cfg,monkeypatch):
    from src.inference import shared_driver_rk4
    original_forward = shared_driver_rk4.decay_forward
    def domain_failure(*args,**kwargs):
        if kwargs.get('decay_multiplier') == 0:
            raise FloatingPointError('invalid physical decay RK stage')
        return original_forward(*args,**kwargs)
    monkeypatch.setattr(shared_driver_rk4,'decay_forward',domain_failure)
    n = 120; operator = np.eye(3*n); driver = np.zeros(n); initial = np.r_[0.,np.ones(4)]
    forward = original_forward(driver,initial,run.volume_drive_parameters(cfg),.25)
    original = dict(driver=driver,initial_state=initial,states=forward['states'],prediction=forward['canonical_prediction'])
    arrays,audits = run.decay_fixed_upstream_replays(cfg,original,operator,np.ones(3))
    assert [audit['status'] for audit in audits] == ['completed','completed','infeasible_counterfactual']
    assert 'replay_slow_prediction' in arrays
    assert 'replay_no_decay_prediction' not in arrays
    assert audits[-1]['identical_upstream_and_hbr'] is None


def test_replay_migration_preserves_parent_and_only_restores_saved_fit(cfg,tmp_path,monkeypatch):
    from src.inference import shared_driver_rk4
    plan = dict(records=[dict(id='fixture',dataset='fixture',subject='s0')],independent_subjects={'fixture':1})
    run.write_json(tmp_path/'plan.json',plan)
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    for relative in ('src/inference/shared_driver_rk4.py','src/inference/t3a_balloon_robust_ssm.py'):
        destination = tmp_path/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((run.CODE_ROOT/relative).read_bytes())
    for stage in ('synthetic','measured'):
        run.write_json(tmp_path/(stage+'_manifest.json'),dict(status='completed_with_software_errors'))
        for spec in run.decay_specs(cfg,plan,stage):
            run.write_json(tmp_path/stage/(spec['key']+'.json'),dict(status='completed',rows=[]))
    spec = run.decay_specs(cfg,plan,'measured')[0]
    n = 120; driver = np.zeros(n); initial = np.r_[0.,np.ones(4)]; operator = np.eye(3*n)
    original_forward = shared_driver_rk4.decay_forward
    forward = original_forward(driver,initial,run.volume_drive_parameters(cfg),.25)
    source = tmp_path/'measured'/(spec['key']+'.json')
    row = dict(arm='original',condition=spec['condition'],qualified=False,converged=True,software_error=True,
        fit_status='exception',integration_passed=True,objective=1.,full_hb_nmse=.25,
        starts=[dict(converged=True,status='completed',objective=1.)],
        error='decay_fixed_upstream_replays\nFloatingPointError: invalid physical decay RK stage\n')
    run.write_json(source,dict(spec=spec,key=spec['key'],status='completed_with_arm_errors',rows=[row]))
    np.savez_compressed(source.with_suffix('.npz'),original_driver=driver,original_initial_state=initial,
        original_states=forward['states'],original_prediction=forward['canonical_prediction'],operator=operator,sd=np.ones(3))
    preserved = {path:path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    def domain_failure(*args,**kwargs):
        if kwargs.get('decay_multiplier') == 0:
            raise FloatingPointError('invalid physical decay RK stage')
        return original_forward(*args,**kwargs)
    monkeypatch.setattr(shared_driver_rk4,'decay_forward',domain_failure)
    monkeypatch.setattr(fitting,'fit_decay_shared_driver',lambda *a,**k:pytest.fail('migration must never refit'))
    migrated = run.decay_replay_migrate(cfg,tmp_path,plan)
    assert migrated['status'] == 'completed' and migrated['corrected_tasks'] == 1
    assert migrated['refits'] == 0 and migrated['fitted_parameters_changed'] == 0
    assert all(path.read_bytes() == content for path,content in preserved.items())
    corrected = json.loads((tmp_path/'evidence_v2'/'measured'/source.name).read_text())
    restored = corrected['rows'][0]
    assert restored['qualified'] and not restored['software_error']
    assert restored['fit_status'] == 'completed' and restored['full_hb_nmse'] == .25
    assert restored['legacy_error'] == row['error']
    assert restored['driver_change_rms_from_original'] == 0
    with np.load(tmp_path/'evidence_v2'/'measured'/source.with_suffix('.npz').name,allow_pickle=False) as arrays:
        assert all(name.startswith('replay_') for name in arrays.files)
    with pytest.raises(ValueError,match='never overwritten'):
        run.decay_replay_migrate(cfg,tmp_path,plan)


def test_worker_isolates_arm_exception_and_retains_nonconvergence(cfg,tmp_path,monkeypatch):
    calls = []
    def fake_fit(target,p,dt,**kwargs):
        calls.append((target.copy(),kwargs))
        if len(calls) == 2:
            raise RuntimeError('fixture arm exception')
        n = len(target); driver = np.zeros(n); initial = np.r_[0.,np.ones(4)]
        coeff = np.zeros(kwargs['forcing_modes'])
        forward = decay_forward(driver,initial,p,dt,decay_multiplier=kwargs['decay_multiplier'],forcing_coefficients=coeff)
        return dict(converged=False,status='failed_numerical',starts=[],evaluations=1,
            prediction=(kwargs['mean_operator']@forward['canonical_prediction'].ravel()).reshape(n,3),
            states=forward['states'],driver=driver,initial_state=initial,forcing_coefficients=coeff,
            objective=1.,scaled_gradient_inf_norm=1.,costs=dict(weighted_data_sse=1.,
                driver_curvature_cost=0.,initial_prior_cost=0.,flow_prior_cost=0.,forcing_prior_cost=0.))
    monkeypatch.setattr(fitting,'fit_decay_shared_driver',fake_fit)
    spec = run.decay_specs(cfg,{},'pilot')[0]
    status = run.decay_worker((cfg,str(tmp_path),str(tmp_path),'pilot',spec))
    task = json.loads((tmp_path/'pilot'/(spec['key']+'.json')).read_text())
    assert status['status'] == 'completed_with_arm_errors'
    assert len(task['rows']) == 6 and len(calls) == 6
    assert task['rows'][1]['software_error']
    assert not any(row['qualified'] for row in task['rows'])
    assert not task['rows'][0]['software_error']
    assert len(calls[0][1]['starts']) == 1
    assert all(len(kwargs['starts']) == 2 for _,kwargs in calls[1:])
    mask = run.volume_drive_mask(120,'joint_full')
    for visible,kwargs in calls:
        assert np.isnan(visible[~mask]).all()
        assert not any(key.startswith('truth') for key in kwargs)
        if kwargs['forcing_modes']:
            assert all(np.array_equal(start['forcing_coefficients'],np.zeros(kwargs['forcing_modes']))
                       for start in kwargs['starts'])
    with np.load(tmp_path/'pilot'/(spec['key']+'.npz'),allow_pickle=False) as arrays:
        assert 'truth_delta' in arrays and 'driven_no_decay_forcing' in arrays


def test_real_solver_runner_short_budget_preserves_all_six_arm_outputs(cfg,tmp_path):
    # A small synthetic software integration check, not a recovery verdict.
    cfg['solver']['max_evaluations'] = 4
    spec = run.decay_specs(cfg,{},'pilot')[0]
    outcome = run.decay_worker((cfg,str(tmp_path),str(tmp_path),'pilot',spec))
    assert outcome['status'] == 'completed'
    task = json.loads((tmp_path/'pilot'/(spec['key']+'.json')).read_text())
    assert len(task['rows']) == 6
    assert not any(row['software_error'] for row in task['rows'])
    assert all(row['integration_passed'] for row in task['rows'])
    with np.load(tmp_path/'pilot'/(spec['key']+'.npz'),allow_pickle=False) as arrays:
        for arm in cfg['decay']['arms']:
            assert arrays[arm['name']+'_prediction'].shape == (120,3)
            assert arrays[arm['name']+'_forcing_coefficients'].size == (7 if arm['input'] else 0)


def test_summary_denominators_include_missing_rows_and_task_failures(cfg,tmp_path):
    plan = dict(records=[dict(id='fixture',dataset='fixture',subject='s0')],independent_subjects={'fixture':1})
    spec = run.decay_specs(cfg,plan,'measured')[0]
    run.write_json(tmp_path/'measured'/(spec['key']+'.json'),dict(status='failed',rows=[]))
    counts = run.decay_summarize(cfg,tmp_path,plan)
    assert counts['measured']['planned'] == 6
    assert counts['measured']['planned_arms'] == 36
    assert counts['measured']['software_failed_tasks'] == 1
    assert counts['measured']['qualified_arms'] == 0
    summary = json.loads((tmp_path/'summary.json').read_text())
    groups = [group for group in summary['groups'] if group['stage']=='measured']
    assert sum(group['planned'] for group in groups) == 36
    assert all(group['qualified'] == 0 for group in groups)
    within_lambda = [pair for pair in summary['paired']
                     if pair['reference']=='slow' and pair['candidate']=='driven_slow']
    assert within_lambda and all(pair['qualified_pairs'] == 0 for pair in within_lambda)
    assert summary['status'] == 'in_progress'


def test_contrast_bootstraps_subjects_after_record_pairing(cfg):
    import pandas as pd
    rows = []
    for subject,multiplier in [('s0',1.),('s1',4.)]:
        for i in range(3 if subject == 's0' else 1):
            for arm,value in [('slow',multiplier),('driven_slow',multiplier/4)]:
                rows.append(dict(subject=subject,id=f'{subject}_{i}',arm=arm,qualified=True,full_hb_nmse=value))
    contrast = run.decay_paired_contrast(cfg,pd.DataFrame(rows),'slow','driven_slow','full_hb_nmse')
    assert contrast['planned_pairs'] == 4 and contrast['subjects'] == 2
    assert contrast['reference_nrmse'] == pytest.approx(np.sqrt(2.5))
    assert contrast['candidate_nrmse'] == pytest.approx(np.sqrt(2.5)/2)
    assert contrast['improvement'] > 0
