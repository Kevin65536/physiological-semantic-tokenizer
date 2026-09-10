import copy
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import yaml

from experiments import evaluate_ssm_overnight_diagnostics as suite
from src.inference import balloon_trajectory_map as batch_map


def inventory(cfg):
    return {subject: [dict(subject=subject, session=session, training_ordinal=i,
        original_ma_trial_position=p, event_index=p, sample_id=f'{subject}/{session}/{p}')
        for session in cfg['sessions'] for i, p in enumerate([0, 1, 2, 3, 5, 6, 7, 8])]
        for subject in cfg['subjects']}


def test_fixed_scope_and_fold_inventory_excludes_original_positions_not_ordinal():
    cfg, *_ = suite.load_config()
    ids = inventory(cfg)
    assert sum(map(len, ids.values())) == 72
    for subject, rows in ids.items():
        suite.validate_identities(rows, cfg, subject)
        assert rows[4]['original_ma_trial_position'] == 5
        for outer in range(4):
            split = suite.folds(rows, outer)
            assert len(split['train']) == 18 and len(split['validation']) == 6
            inner_val = []
            for inner in split['inner']:
                assert not set(inner['train']) & set(inner['validation'])
                assert not set(inner['validation']) & set(split['validation'])
                inner_val += inner['validation']
            assert sorted(inner_val) == sorted(split['train'])
    broken = copy.deepcopy(ids['subject_01'])
    broken[4]['original_ma_trial_position'] = 4
    with pytest.raises(ValueError, match='boundary'):
        suite.validate_identities(broken, cfg, 'subject_01')


def test_task_table_counts_fairness_and_dependency_layers_are_fixed():
    cfg, *_ = suite.load_config()
    tasks = suite.make_tasks(cfg, inventory(cfg))
    assert tasks == suite.make_tasks(cfg, inventory(cfg))
    ids = {t['id']: t for t in tasks}
    assert len(ids) == len(tasks)
    for family in suite.FAMILIES:
        core = sum(t['planned_solves'] for t in tasks if t['family'] == family and t['layer'] < 3)
        assert core <= cfg['budget']['core_solve_caps'][family]
    for t in tasks:
        assert all(ids[d]['layer'] <= t['layer'] for d in t['dependencies'])
    temporal = [t for t in tasks if t['kind'] == 'temporal']
    assert len(temporal) == 3168
    assert Counter(t['solver'] for t in temporal) == {'O0': 1056, 'O1': 1056, 'O2': 1056}
    assert set(t['family'] for t in tasks[:24]) == set(suite.FAMILIES)


def test_mean_gain_and_independent_process_sensitivities_preserve_noise_and_physiology():
    _, _, base, _, _ = suite.load_config()
    p, c, spec = suite.model(base, suite.BASE)
    changed, _, gain = suite.model(base, dict(suite.BASE, gain=1.5, sigma_h=.25, sigma_r=1.))
    np.testing.assert_allclose(changed.fixed.process_std, np.asarray(p.fixed.process_std)*[1., .25, .25, .25, .25, .25])
    np.testing.assert_array_equal(spec.effective_noise_scale, gain.effective_noise_scale)
    assert changed.fixed.P0 == p.fixed.P0 and changed.fixed.Q0 == p.fixed.Q0
    z = np.array([.01, .02, .02, -.01, .03, .01])
    physical = suite.core.transformed_to_physical(z)
    np.testing.assert_allclose(suite.core.observation_map(physical, changed, gain),
                              suite.core.observation_map(physical, p, spec)*[1, 1.5, 1.5])
    batch = suite.joint._batch_observation(z[None], changed, gain)[0]
    np.testing.assert_allclose(batch, suite.core.observation_map(physical, changed, gain))
    _, _, noisy = suite.model(base, dict(suite.BASE, noise=2.))
    np.testing.assert_allclose(noisy.effective_noise_scale, spec.effective_noise_scale*[1, 2, 2])
    with pytest.raises(ValueError, match='fnirs_gain'):
        replace(spec, fnirs_gain=0).validate()


def test_gw_unlocks_process_gain_without_moving_damping_noise_or_measurement_mapping():
    cfg, _, base, _, _ = suite.load_config()
    original, c, spec = suite.model(base, suite.BASE)
    changed, _, changed_spec = suite.model(base, dict(suite.BASE, g=.3, w=-.25))
    assert changed.fixed.neurovascular_gain/changed.fixed.gamma == pytest.approx(
        original.fixed.neurovascular_gain/original.fixed.gamma*np.exp(.3))
    assert changed.free.kappa/(2*np.sqrt(changed.fixed.gamma)) == pytest.approx(
        original.free.kappa/(2*np.sqrt(original.fixed.gamma)))
    assert changed.free.tau == original.free.tau
    assert changed.fixed.process_std == original.fixed.process_std
    assert changed_spec == spec
    z = np.array([.01, .02, .03, .01, .02, -.01])
    np.testing.assert_array_equal(suite.core.observation_map(suite.core.transformed_to_physical(z), original, spec),
                                  suite.core.observation_map(suite.core.transformed_to_physical(z), changed, changed_spec))
    assert not np.allclose(suite.core.rk4_transition(z, original, c), suite.core.rk4_transition(z, changed, c))
    with pytest.raises(ValueError, match='registered support'):
        suite.model(base, dict(suite.BASE, g=.600001))
    assert all(c['gain'] == 1 for c in suite.candidates(cfg, 'N7'))


def test_gw_rules_reuse_identical_inner_fits_and_cannot_select_from_wrong_axis(tmp_path):
    cfg, *_ = suite.load_config()
    tasks = suite.make_tasks(cfg, inventory(cfg))
    choices = [t for t in tasks if t['family'] == 'N7' and t['kind'] == 'select'
               and t['subject'] == 'subject_01' and t['outer'] == 0]
    assert {t['rule'] for t in choices} == {'GW', 'W_only', 'G_only'}
    panel = {c['id']: c for c in suite.candidates(cfg, 'N7')}
    full = next(t for t in choices if t['rule'] == 'GW')
    for name, identifiers in full['inner_cells'].items():
        c = panel[name]
        loss = .1 if (c['g'], c['w']) == (.3, .25) else .5 if (c['g'], c['w']) == (.3, 0) else .7 if (c['g'], c['w']) == (0, -.25) else 1.
        block = dict(status='completed', rows=[dict(status='completed', mode='center_EEG' if i % 2 == 0 else 'center_fNIRS',
                     center_metrics={m: dict(nmse=loss) for m in suite.MODALITIES}) for i in range(12)])
        for identifier in identifiers:
            suite.atomic_json(tmp_path/'cells'/identifier/'result.json', block)
    expected = {'GW': (.3, .25), 'W_only': (0., -.25), 'G_only': (.3, 0.)}
    for task in choices:
        assert all(ids == full['inner_cells'][name] for name, ids in task['inner_cells'].items())
        selected = suite.select_candidate(tmp_path, cfg, task)['selected']
        assert (selected['g'], selected['w']) == expected[task['rule']]
    for rule in expected:
        assert sum(t['family'] == 'N7' and t['kind'] == 'selected_outer' and t['rule'] == rule for t in tasks) == 72
    for task in tasks:
        if task['family'] == 'N7' and task['kind'] == 'synthetic':
            assert task['candidate']['g'] == 0 or task['candidate']['gain'] == 1


def test_synthetic_measurement_gain_retains_driver_and_noise_draw():
    _, _, base, _, _ = suite.load_config()
    original = suite.generated_trial(base, 'nonlinear_student_t', 817, 32, truth_g=.3, truth_w=-.25)
    changed = suite.generated_trial(base, 'nonlinear_student_t', 817, 32, truth_g=.3, truth_w=-.25, truth_gain=.75)
    np.testing.assert_array_equal(changed['states'], original['states'])
    np.testing.assert_allclose(changed['clean'], original['clean']*[1., .75, .75], atol=1e-14)
    np.testing.assert_allclose(changed['observations']-changed['clean'], original['observations']-original['clean'], atol=1e-14)


def test_gw_improvement_is_measured_against_fitted_w_rule_not_only_fixed_reference():
    cfg, *_ = suite.load_config()
    tasks, results = [], {}
    for family, rule, nmse in [('N1', 'fixed', 1.), ('N7', 'W_only', .5), ('N7', 'GW', .6)]:
        for subject, identities in inventory(cfg).items():
            for i, identity in enumerate(identities):
                key = f'{family}_{rule}_{subject}_{i}'
                tasks.append(dict(id=key, family=family, rule=rule, subject=subject, trial=i,
                    kind='outer' if family == 'N1' else 'selected_outer', candidate=suite.BASE))
                results[key] = dict(status='completed', selected_candidate=suite.BASE, rows=[dict(
                    mode=mode, status='completed', subject=subject, session=identity['session'], sample_id=identity['sample_id'],
                    center_metrics={m: dict(nmse=nmse) for m in suite.MODALITIES}) for mode in suite.OUTER_MASKS])
        if family == 'N7':
            for fold in range(12):
                key = f'{rule}_select_{fold}'
                tasks.append(dict(id=key, family=family, kind='select', rule=rule))
                results[key] = dict(status='completed', selected=suite.BASE)
    row = next(r for r in suite.candidate_summary(cfg, tasks, results) if r['family'] == 'N7' and r['rule'] == 'GW')
    assert row['comparison_reference'] == 'N7/W_only'
    assert row['status'] == 'complete_outer_rule'
    assert row['relative_risk_improvement'] == pytest.approx(-.2)
    assert not row['measured_screen_passed']


def test_engineering_thresholds_include_native_hidden_intervention_and_density():
    cfg, dc, base, _, _ = suite.load_config()
    result = suite.engineering_checks(cfg, dc, base)
    assert all(item['passed'] for item in result.values())
    assert result['O2']['measured']['hidden_intervention'] == 0


def test_eog_regression_fits_training_only_and_hides_auxiliary_before_cleaning():
    rng = np.random.default_rng(43)
    auxiliary = rng.normal(size=(4, 6000, 2))
    raw = rng.normal(size=(4, 6000, 3))+auxiliary@np.array([[1., 0., .3], [.1, 1., .2]])
    fit = suite.eog_coefficients(raw, auxiliary, [0, 1])
    changed, changed_aux = raw.copy(), auxiliary.copy()
    changed[2:] += 1e5
    changed_aux[2:] *= 1e5
    other = suite.eog_coefficients(changed, changed_aux, [0, 1])
    for a, b in zip(fit, other):
        np.testing.assert_array_equal(a, b)
    before = suite.cleaned_eeg(raw[2], auxiliary[2], *fit, True)
    changed = raw[2].copy()
    changed[2600:3400] = 1e80
    after = suite.cleaned_eeg(changed, auxiliary[2], *fit, True)
    np.testing.assert_array_equal(before, after)


def test_map_failure_does_not_borrow_variance_or_clip_initial_path():
    _, dc, base, _, _ = suite.load_config()
    p, c, _ = suite.model(base, suite.BASE)
    op = suite.repair.trajectory_operator(8, 'model', dc)
    objective = batch_map.TrajectoryObjective(np.zeros((8, 3)), p, c, op, [.01]*3)
    invalid = np.zeros((8, 6))
    invalid[:, 2] = -100
    fit = batch_map._optimize(objective, invalid, 2)
    assert not fit['converged'] and fit['status'] == 'infeasible_initial'
    assert fit['evaluations'] <= 2
    with pytest.raises(ValueError, match='max_nfev'):
        batch_map.smooth_balloon_trajectory_map(np.zeros((8, 3)), p, config=c,
            trajectory_spec=op, noise_variance=[.01]*3, max_nfev=201)


def test_noisy_target_conditioning_keeps_shared_noise_instead_of_adding_independent_noise():
    _, dc, base, _, _ = suite.load_config()
    p, c, _ = suite.model(base, suite.BASE)
    g = suite.repair.generate_linear_gaussian(base, 8, 820)
    op = suite.repair.trajectory_operator(8, 'model', dc)
    noise = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
    mask = np.ones((8, 3), dtype=bool)
    mean, var = suite.gaussian_noisy_target(g['observations'], p, c, op, noise, mask, 1e-10)
    np.testing.assert_allclose(mean, g['observations'], atol=1e-10)
    np.testing.assert_allclose(var, 0., atol=1e-12)
    mask[3:5, 0] = False
    masked = op.with_visible_interpolation(mask, mask)
    mean, var = suite.gaussian_noisy_target(masked.apply(g['observations'], mask), p, c, masked, noise, mask, 1e-10)
    assert np.all(var[3:5, 0] > 0)
    np.testing.assert_allclose(mean[mask], g['observations'][mask], atol=1e-10)


def test_memory_shortage_does_not_force_one_worker(monkeypatch):
    cfg, *_ = suite.load_config()
    monkeypatch.setattr(suite, 'available_memory', lambda: 100)
    value = suite.resource_budget(cfg, [dict(family='N1', peak_rss_bytes=1000)])
    assert value['workers'] == 0 and value['status'] == 'insufficient_memory'


def test_worker_alarm_isolates_timeout_and_keeps_result(tmp_path, monkeypatch):
    cfg, *_ = suite.load_config()
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    def slow(*args):
        time.sleep(.2)
    monkeypatch.setattr(suite, 'linear_cell', slow)
    result = suite.execute_cell(tmp_path, cfg, dict(id='bounded', family='N1', kind='linear', timeout_seconds=.02))
    assert result['status'] == 'timeout'
    assert suite.read_json(tmp_path/'cells/bounded/result.json')['status'] == 'timeout'
    # The alarm is cancelled; subsequent work in the same worker remains usable.
    time.sleep(.03)


def test_incomplete_candidate_selection_cannot_win_by_dropping_failed_trials(tmp_path):
    cfg, *_ = suite.load_config()
    def block(nmse, failed=False):
        return dict(status='completed', rows=[dict(status='failed_domain' if failed and i == 0 else 'completed',
            mode='center_EEG' if i % 2 == 0 else 'center_fNIRS',
            center_metrics={m: dict(nmse=nmse) for m in suite.MODALITIES}) for i in range(12)])
    for i in range(3):
        suite.atomic_json(tmp_path/f'cells/b{i}/result.json', block(1.))
        suite.atomic_json(tmp_path/f'cells/c{i}/result.json', block(.01, failed=i == 0))
    task = dict(family='N3', inner_cells={'baseline': ['b0', 'b1', 'b2'], 'noise_0.5': ['c0', 'c1', 'c2']})
    selected = suite.select_candidate(tmp_path, cfg, task)
    assert selected['selected']['id'] == 'baseline'
    assert selected['candidates']['noise_0.5']['risk'] is None
    assert selected['candidates']['noise_0.5']['completed_fits'] == 35


def test_complete_synthetic_and_temporal_worker_routes(tmp_path):
    cfg, dc, base, _, _ = suite.load_config()
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    suite.atomic_json(tmp_path/'preflight.json', suite.engineering_checks(cfg, dc, base))
    for solver in ('O0', 'O1', 'O2'):
        task = dict(id='test_'+solver, kind='temporal', family='N2', law='nonlinear_gaussian',
                    seed=cfg['seed']+11, variant='model', mode='full', w=0., solver=solver)
        result = suite.execute_cell(tmp_path, cfg, task)
        assert result['status'] == 'completed', suite.cell_result(tmp_path, task['id'])
        row = suite.cell_result(tmp_path, task['id'])['rows'][0]
        assert row['truth']['r']['nrmse'] is not None
    task = dict(id='synthetic_route', kind='synthetic', family='N3', condition='matched_student_t',
                seed=cfg['seed']+10, stream='discovery', candidate=suite.BASE)
    result = suite.execute_cell(tmp_path, cfg, task)
    assert result['status'] == 'completed', suite.cell_result(tmp_path, task['id'])
    assert len(suite.cell_result(tmp_path, task['id'])['rows']) == 3


def test_montage_neighbors_are_selected_without_response_values(tmp_path, monkeypatch):
    geometry = tmp_path/'channel_geometry'
    geometry.mkdir()
    base = dict(dataset_id='eeg_fnirs_single_trial', canonical_subject_id='subject_01',
                coordinate_system='native', coordinate_units='normalized', y=0., z=0.)
    rows = [dict(base, modality='fnirs', channel_name='pair', x=0.)]
    rows += [dict(base, modality='eeg', channel_name=name, x=x) for name, x in [('F3', 3.), ('Cz', 1.), ('Pz', 2.), ('F4', 4.)]]
    import json
    (geometry/'channels.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    detail = dict(subject='subject_01', eeg_channels=['F3', 'Cz', 'Pz', 'F4'], fnirs_pairs=['pair'])
    indices, report = suite.local_channels(detail, 0, {'data': {'cache_root': str(tmp_path)}})
    assert indices == [1, 2, 0]
    assert report['status'] == 'native_montage_nearest_eeg'


def test_hard_budget_stops_worker_and_writes_complete_denominator_report(tmp_path, monkeypatch):
    import signal
    cfg, *_ = suite.load_config()
    cfg['budget']['hours'] = .00001
    cfg['budget']['max_workers'] = 1
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    suite.atomic_json(tmp_path/'manifest.json', dict(execution='ready'))
    suite.atomic_json(tmp_path/'frozen_input_identity.json', {})
    suite.atomic_json(tmp_path/'pilots.json', dict(pilots=[dict(family='N1', peak_rss_bytes=1024**3)], preparations=[]))
    task = dict(id='pending', family='N2', kind='measured_map', layer=1, queue_position=0, planned_solves=0,
                timeout_seconds=900, dependencies=[], reason='no native data in this test')
    suite.persist_tasks(tmp_path, [task])
    monkeypatch.setattr(suite, 'verify_snapshot', lambda path: None)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        suite.run_scheduler(tmp_path, cfg)
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)
    result = suite.read_json(tmp_path/'manifest.json')
    assert result['execution'] == 'stopped_budget'
    assert suite.cell_result(tmp_path, 'pending')['status'] in ('timeout', 'not_started_budget')
    assert (tmp_path/'OVERNIGHT_REPORT.md').exists()
    assert 'pending' in (tmp_path/'failure_attribution.csv').read_text()
