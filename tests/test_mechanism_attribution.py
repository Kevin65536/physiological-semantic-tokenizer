"""Known-truth pairing, recovery, profile semantics and public-table boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.mechanism_attribution import (
    make_worlds, training_scales, fit_profile, select_profile, recovery_metrics,
    signed_projection, paired_counterfactual_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mechanism_runner',
    ROOT/'experiments/scripts/evaluate_mechanism_attribution.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def cfg():
    config = runner.read_config(runner.DEFAULT_CONFIG)
    config['tensor']['steps'] = 24
    config['synthetic']['training_seeds'] = [101, 102]
    return config


def test_factorial_changes_only_declared_mechanisms(cfg):
    scales = training_scales(cfg)
    worlds = make_worlds(cfg, 201, .2, scale=scales['observation'])
    assert len(worlds) == 8
    for w in worlds:
        np.testing.assert_array_equal(w['driver'], worlds[0]['driver'])
        np.testing.assert_array_equal(w['initial_state'], worlds[0]['initial_state'])
        np.testing.assert_array_equal(w['noise'], worlds[0]['noise'])
        np.testing.assert_allclose(w['observation'],
            w['physical']*np.array([1, w['gain'], w['gain']])+w['component']+w['noise'])
        if w['tau'] == 2:
            np.testing.assert_array_equal(w['states'], worlds[0]['states'])
        if not w['common_present']:
            assert not w['component'].any()
    assert not np.allclose(worlds[0]['states'], worlds[-1]['states'])
    assert worlds[-1]['component'][:, 0].max() == 0
    np.testing.assert_allclose(worlds[-1]['component'][:, 1]*.35,
                               worlds[-1]['component'][:, 2]*.65)


def test_scales_are_training_only_and_oracle_recovers_changed_tau(cfg):
    scales = training_scales(cfg)
    changed = deepcopy(cfg)
    changed['synthetic']['evaluation_seeds'] = [9999]
    for k in scales:
        np.testing.assert_array_equal(scales[k], training_scales(changed)[k])
    world = make_worlds(cfg, 201, 0, scale=scales['observation'])[4]
    nodes = fit_profile(world['observation'], cfg, scales['observation'], .015,
        oracle_driver=world['driver'], oracle_initial=world['initial_state'])
    selected = select_profile(nodes)
    assert selected['tau'] == world['tau'] == 3
    metrics = recovery_metrics(world, selected, scales)
    assert metrics['physical_Hb_nrmse'] == pytest.approx(0, abs=1e-14)
    assert metrics['driver_nrmse'] == 0
    assert metrics['state_nrmse'] == 0


def test_profile_requires_convergence_and_objective_is_not_data_sse():
    nodes = [dict(tau=2., converged=True, objective=2., weighted_data_sse=1.),
             dict(tau=3., converged=True, objective=3., weighted_data_sse=0.),
             dict(tau=4., converged=False, objective=.01, weighted_data_sse=0.)]
    assert select_profile(nodes)['tau'] == 2
    assert select_profile(nodes, 3.)['tau'] == 3
    assert select_profile(nodes, 4.) is None


def test_nonlinear_profile_refits_driver_and_five_initial_coordinates(cfg):
    scales = training_scales(cfg)
    world = make_worlds(cfg, 202, 0, scale=scales['observation'])[0]
    cfg['fit']['tau_grid_s'] = [2.]
    nodes = fit_profile(world['observation'], cfg, scales['observation'], .015, common=True)
    fit = nodes[0]
    assert fit['converged']
    assert fit['driver'].shape == world['driver'].shape
    assert fit['initial_state'].shape == (5,)
    assert fit['total_free_parameters'] == cfg['tensor']['steps']+5+4
    assert np.linalg.norm(fit['driver']) > 0
    np.testing.assert_allclose(fit['prediction'],
                               fit['physical_prediction']+fit['observation_component'])
    assert fit['weighted_data_sse'] <= fit['objective']
    initial_cost = cfg['fit']['initial_penalty']*np.sum((fit['initial_state']-np.r_[0., np.ones(4)])**2)
    assert fit['objective'] == pytest.approx(fit['weighted_data_sse']+
        cfg['fit']['penalty']*fit['roughness']+initial_cost+fit['observation_component_cost'])


def test_signed_projection_is_not_energy_fraction_and_truth_preservation(cfg):
    reference = np.array([[1., -1.], [2., -2.]])
    assert signed_projection(-2*reference, reference, np.ones(2)) == -2
    assert np.isnan(signed_projection(reference, np.zeros_like(reference), np.ones(2)))
    scales = training_scales(cfg)
    worlds = make_worlds(cfg, 201, 0, scale=scales['observation'])
    def oracle(world):
        return dict(tau=world['tau'], physical_prediction=world['physical'],
                    observation_component=world['component'], states=world['states'])
    result = paired_counterfactual_metrics(worlds[4], worlds[0],
                                           oracle(worlds[4]), oracle(worlds[0]), scales)
    assert result['delta_tau_error_s'] == 0
    assert result['delta_physical_Hb_nrmse'] == 0
    assert result['tau_change_extra_projection'] == 0


def test_contract_denominator_and_fail_closed_boundaries(tmp_path):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    plan = runner.make_plan(cfg)
    assert plan['planned_seed_noise_tasks'] == 24
    assert plan['planned_endpoints'] == 1536
    assert plan['planned_profile_nodes'] == 5760
    for field, value in [('protected_data', 'allowed'), ('data_boundary', 'raw'),
                         ('artifact_root', '/tmp/other'), ('experiment_id', 'other')]:
        changed = deepcopy(cfg)
        changed[field] = value
        path = tmp_path/'bad.yaml'
        path.write_text(runner.yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            runner.read_config(path)
    changed = deepcopy(cfg)
    changed['synthetic']['training_seeds'] = [201]
    path = tmp_path/'leaked.yaml'
    path.write_text(runner.yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='disjoint'):
        runner.read_config(path)


def test_retained_strata_include_zero_event_windows_and_training_threshold(tmp_path, monkeypatch):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    obs = tmp_path/cfg['retained']['observation_run']
    attr = tmp_path/cfg['retained']['attribution_run']
    obs.mkdir(parents=True)
    attr.mkdir(parents=True)
    (obs/'summary.json').write_text(json.dumps(dict(execution='completed', experiment_id='OBSERVATION-SEMANTICS-v1')))
    (attr/'summary.json').write_text(json.dumps(dict(execution='completed', schema='shared_driver_attribution_summary_v1')))
    records, events, metrics = [], [], []
    for subject, split, counts in [('subject_01', 'train', [0, 2]), ('subject_02', 'evaluate', [0, 4]),
                                   ('subject_04', 'evaluate', [0, 6])]:
        windows = [dict(id=f'{subject}_{i}') for i in range(2)]
        records.append(dict(dataset='eeg_fnirs_single_trial', subject=subject, split=split, windows=windows))
        for window, count in zip(windows, counts):
            events += [dict(dataset='eeg_fnirs_single_trial', subject=subject, id=window['id'], split=split)]*count
            if split == 'evaluate':
                for arm in ['EOG_real', 'training_mean', 'EOG_wrong_subject', 'EOG_time_shift']:
                    metrics.append(dict(dataset='eeg_fnirs_single_trial', subject=subject, id=window['id'],
                        arm=arm, endpoint='HbT', nrmse=1. if arm == 'EOG_real' else 2.))
    (obs/'measured_plan.json').write_text(json.dumps(dict(records=records,
        split='odd_native_subject_number_train_even_evaluate')))
    pd.DataFrame(events).to_csv(obs/'event_inventory.csv', index=False)
    pd.DataFrame(metrics).to_csv(obs/'event_metrics.csv', index=False)
    pd.DataFrame([dict(dataset='fixture', mean_gain=0.)]).to_csv(attr/'anchor_paired.csv', index=False)
    audit_path = tmp_path/cfg['retained']['source_audit']
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps(dict(records=[dict(subject='s1', systemic_auxiliary_candidates=[])],
        inspected_files=[], archive_subject_members=[], independent_calibration_established=False,
        missing_semantics='not_found_in_selected_release')))
    out = tmp_path/'output'
    out.mkdir()
    result = runner.retained_analysis(cfg, out)
    assert result['training_thresholds']['eeg_fnirs_single_trial'] == 1
    inventory = pd.read_csv(out/'retained_event_burden_inventory.csv')
    assert len(inventory) == 6
    assert (inventory.event_count == 0).sum() == 3
    summary = pd.read_csv(out/'retained_EOG_burden.csv')
    assert set(summary.burden) == {'low', 'high'}
    assert (summary.paired_windows == 2).all()
    assert (summary.independent_units == 2).all()
    assert (summary['mean'] == 1).all()


def test_task_exceptions_keep_every_stratum_denominator(tmp_path):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    plan = runner.make_plan(cfg, pilot=True)
    for task in plan['tasks']:
        runner.write_json(tmp_path/'tasks'/task['key']/'status.json',
                          dict(status='failed_task', task=task, error='injected failure'))
    result = runner.summarize(cfg, tmp_path, plan)
    assert result['execution'] == 'completed'
    assert result['failed_tasks'] == 2
    assert result['terminal_endpoints'] == result['planned_endpoints'] == 128
    assert result['profile_nodes'] == 480
    assert result['succeeded_endpoints'] == 0
    endpoint_summary = pd.read_csv(tmp_path/'endpoint_summary.csv')
    paired = pd.read_csv(tmp_path/'paired_comparisons.csv')
    assert (endpoint_summary.planned_seeds == 1).all()
    assert (endpoint_summary.successful_fits == 0).all()
    assert (paired.planned_pairs == 1).all()
    assert (paired.common_success == 0).all()
    runner.write_json(tmp_path/'plan.json', plan)
    runner.write_json(tmp_path/'calibration.json', dict(training_seeds=[101, 102],
        scales=dict(observation=[1, 1, 1], driver=1, state=[1]*5)))
    verification = runner.verify(cfg, tmp_path)
    assert verification['status'] == 'passed'
    assert verification['endpoint_rows'] == 128
    assert verification['profile_rows'] == 480


def test_single_node_exception_preserves_other_profile_nodes(cfg, monkeypatch):
    import src.inference.mechanism_attribution as mechanism
    def failing_solver(observation, parameters, dt, **kwargs):
        if parameters.free.tau == 2:
            raise FloatingPointError('injected node failure')
        return dict(status='completed', converged=True, objective=parameters.free.tau)
    monkeypatch.setattr(mechanism, 'fit_nonlinear_shared_driver', failing_solver)
    nodes = fit_profile(np.zeros((24, 3)), cfg, np.ones(3), .015)
    assert len(nodes) == 5
    assert nodes[2]['status'] == 'failed_node_exception'
    assert sum(n['converged'] for n in nodes) == 4
    assert select_profile(nodes)['tau'] == 1


def test_running_task_partial_tables_are_not_consumed(tmp_path):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    plan = runner.make_plan(cfg, pilot=True)
    task = plan['tasks'][0]
    out = tmp_path/'tasks'/task['key']
    runner.write_json(out/'status.json', dict(status='running', task=task))
    (out/'endpoints.csv').write_text('partially written invalid table')
    result = runner.summarize(cfg, tmp_path, plan)
    assert result['execution'] == 'incomplete'
    assert result['planned_endpoints'] == result['endpoint_rows'] == 128
    assert result['terminal_endpoints'] == 0
    assert result['status_counts'] == {'pending': 128}


def test_posthoc_projection_excess_removes_genuine_component_overlap(tmp_path):
    spec = importlib.util.spec_from_file_location('mechanism_renderer',
        ROOT/'experiments/scripts/render_mechanism_attribution.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    runner.write_json(tmp_path/'summary.json', dict(execution='completed',
        resolved_reference_parameters=dict(free=dict(tau=2.))))
    rows = []
    for tau in [2., 3.]:
        for seed in [201, 202]:
            for arm, value in [('oracle_mismatch', .4), ('free_common', .7),
                               ('fixed_common', .5), ('oracle_driver_common', .4)]:
                rows.append(dict(key=str(seed), seed=seed, noise_multiplier=.2, world=str(tau),
                    tau_true_s=tau, hb_gain=1., common_present=True, arm=arm, converged=True,
                    tau_change_extra_projection=value if tau == 3 else np.nan))
    pd.DataFrame(rows).to_csv(tmp_path/'endpoints.csv', index=False)
    metadata = renderer.projection_excess_audit(tmp_path)
    result = pd.read_csv(tmp_path/'component_projection_excess_summary_v1.csv')
    assert not result[result.tau_true_s == 2].contrast_defined.any()
    assert result[result.tau_true_s == 2]['mean'].isna().all()
    changed = result[result.tau_true_s == 3].set_index('arm')
    assert changed.loc['free_common', 'mean'] == pytest.approx(.3)
    assert changed.loc['oracle_driver_common', 'mean'] == 0
    assert metadata['status'].startswith('posthoc_')
    with pytest.raises(ValueError, match='exists'):
        renderer.projection_excess_audit(tmp_path)
