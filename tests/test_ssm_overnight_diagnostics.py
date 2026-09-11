import copy
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from src.metrics.trajectory_reliability import canonical_residual_fields
import yaml

from experiments import evaluate_ssm_overnight_diagnostics as suite
from src.inference import balloon_trajectory_map as batch_map


def test_import_keeps_source_roots_separate_from_external_data_root(tmp_path):
    import os
    import subprocess
    import sys

    data_root = tmp_path/'recordings'
    code = '''
import os
import sys
from pathlib import Path
thread_settings = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
from src.inference import observation_baselines
assert not any(name.startswith('experiments') for name in sys.modules)
assert 'matplotlib.pyplot' not in sys.modules
assert thread_settings == {name: os.environ.get(name) for name in thread_settings}
from experiments import evaluate_ssm_overnight_diagnostics as suite

assert suite.ROOT == Path(os.environ['SSM_PROJECT_ROOT'])
for module in (suite.step5, suite.diagnostic, suite.repair):
    assert module.ROOT == suite.CODE_ROOT
suite.load_config(suite.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml')
suite.repair.load_config()
'''
    subprocess.run([sys.executable, '-c', code], cwd=suite.CODE_ROOT, check=True,
                   env={**os.environ, 'SSM_PROJECT_ROOT': str(data_root)})


def test_native_preparation_passes_data_root_at_loader_boundary(tmp_path, monkeypatch):
    cfg, dc, base, measured, metadata = suite.load_config()
    monkeypatch.setattr(suite, 'ROOT', tmp_path)

    class LoaderReached(Exception):
        pass

    def load(subject, *args, data_root=None, **kwargs):
        assert subject == cfg['subjects'][0]
        assert data_root == tmp_path
        assert kwargs['retain_native'] is True
        raise LoaderReached

    monkeypatch.setattr(suite.diagnostic, 'load_training_subject', load)
    with pytest.raises(LoaderReached):
        suite.prepare_subject(tmp_path/'run', cfg, dc, base, measured, metadata,
                              cfg['subjects'][0], [])


def inventory(cfg):
    return {subject: [dict(subject=subject, session=session, training_ordinal=i,
        original_ma_trial_position=p, event_index=p, sample_id=f'{subject}/{session}/{p}')
        for session in cfg['sessions'] for i, p in enumerate([0, 1, 2, 3, 5, 6, 7, 8])]
        for subject in cfg['subjects']}


def test_retained_residual_fields_read_without_changing_evidence(tmp_path):
    import csv
    import json
    from experiments.scripts import render_ssm_overnight_report as renderer

    saved = dict(status='completed', rows=[dict(
        task_id='fixture', innovation_structure={'EEG': {'acf': [1., .25]}},
        process_innovation_rms=[.01]*6, standardized_process_innovation_rms=[.1]*6,
        process_innovation_status='completed')])
    expected = dict(status='completed', rows=[dict(
        task_id='fixture', predictive_residual_structure={'EEG': {'acf': [1., .25]}},
        state_transition_residual_rms=[.01]*6, standardized_state_transition_residual_rms=[.1]*6,
        state_transition_residual_status='completed')])
    path = tmp_path/'result.json'
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    assert suite.read_json(path) == renderer.read_json(path) == expected
    assert path.read_bytes() == before
    assert canonical_residual_fields(expected) == expected

    path = tmp_path/'trial_metrics.csv'
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(saved['rows'][0]))
        writer.writeheader()
        writer.writerow({key: json.dumps(value) for key, value in saved['rows'][0].items()})
    before = path.read_bytes()
    assert renderer.read_csv(path, renderer.KEEP) == [{
        'task_id': 'fixture', 'predictive_residual_structure': {'EEG': {'acf': [1., .25]}}}]
    assert path.read_bytes() == before


def test_predictive_residual_schema_preserves_values_and_rejects_ambiguous_fields():
    saved = dict(rows=[dict(standardized_innovations={'EEG': {'rms': 2.5}})])
    expected = dict(rows=[dict(standardized_predictive_residuals={'EEG': {'rms': 2.5}})])
    assert canonical_residual_fields(saved) == expected
    both = dict(standardized_innovations=[1.], standardized_predictive_residuals=[2.])
    with pytest.raises(ValueError, match='duplicate residual field'):
        canonical_residual_fields(both)


def test_explicit_rectangular_noise_matches_dense_gaussian_conditioning():
    """The mean clock and feature-noise clock have different dimensions."""
    _, dc, base, _, _ = suite.load_config()
    p, c, _ = suite.model(base, suite.BASE)
    n = 8
    op = suite.repair.trajectory_operator(n, 'model', dc)
    rng = np.random.default_rng(167)
    factor = rng.normal(size=(3*n, 5*n))*.01
    y = rng.normal(size=(n, 3))*.01
    ref = suite.joint.smooth_balloon_trajectory_reference(
        y, p, config=c, trajectory_spec=op, noise_variance=None, noise_factor=factor)
    prior = suite.joint.linearized_trajectory_prior(n, p, c)
    h = np.kron(np.eye(n), suite.core.observation_jacobian(np.zeros(6), p))
    offset = np.tile(suite.core.observation_map(np.array([0., 0., 1., 1., 1., 1.]), p), n)
    cross = prior@h.T
    expected = cross@np.linalg.solve(h@cross+factor@factor.T, y.ravel()-offset)
    np.testing.assert_allclose(ref['transformed_mean'].ravel(), expected, atol=1e-11)
    fitted = batch_map.smooth_balloon_trajectory_map(
        y, p, config=c, trajectory_spec=op, noise_variance=None, noise_factor=factor, linear=True)
    assert fitted['status'] == 'completed'
    np.testing.assert_allclose(fitted['transformed_mean'].ravel(), expected, atol=1e-8)
    with pytest.raises(ValueError, match='one noise owner'):
        batch_map.TrajectoryObjective(y, p, c, op, [.01]*3, noise_factor=factor)
    _, _, observation = suite.model(base, suite.BASE)
    scaled = observation.reexpress([1.3, .7, 1.8])
    changed = suite.joint.smooth_balloon_trajectory_reference(
        y*scaled.coordinate_scale, p, config=c, trajectory_spec=op, noise_variance=None,
        noise_factor=factor*np.tile(scaled.coordinate_scale, n)[:, None], observation_spec=scaled)
    np.testing.assert_allclose(changed['transformed_mean'], ref['transformed_mean'], atol=1e-10)
    assert changed['parameter_log_likelihood']-ref['parameter_log_likelihood'] == pytest.approx(
        -n*np.log(scaled.coordinate_scale).sum(), abs=1e-8)


def test_native_feature_boundary_recomposes_and_preserves_existing_output():
    from scipy.signal import resample_poly
    from src.data.homer2_preprocessing import bandpass_fnirs
    rng = np.random.default_rng(178)
    eeg = rng.normal(size=(6000, 3))
    intensity = np.exp(rng.normal(size=(300, 2, 2))*.02)
    original = suite.step5.preprocess_native_trial(eeg, intensity)
    exposed = suite.step5.preprocess_native_trial(eeg, intensity, retain_feature_boundary=True)
    np.testing.assert_array_equal(exposed['fnirs'], original['fnirs'])
    np.testing.assert_array_equal(exposed['eeg_log_power'], original['eeg_log_power'])
    boundary = exposed['feature_boundary']
    filtered, _ = bandpass_fnirs(boundary['fnirs'], sample_rate_hz=10.)
    replay = resample_poly(filtered, 2, 5, axis=0)
    np.testing.assert_allclose(replay, original['fnirs'], rtol=1e-6, atol=1e-10)
    assert boundary['eeg_time'].shape == (120,) and boundary['fnirs_time'].shape == (300,)
    with pytest.raises(ValueError, match='before feature masking'):
        suite.step5.preprocess_native_trial(eeg, intensity, mask_name='center_EEG', retain_feature_boundary=True)


def v3_config():
    return suite.load_config(suite.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml')


def test_v3_prerequisite_prevents_native_access(tmp_path, monkeypatch):
    cfg, dc, base, measured, metadata = v3_config()
    def forbidden(*args, **kwargs):
        raise AssertionError('native loader crossed an unmet synthetic prerequisite')
    monkeypatch.setattr(suite.diagnostic, 'load_training_subject', forbidden)
    task = dict(kind='v3_native', subject='subject_01', prerequisites=['v3_S1_gate'])
    result = suite.v3_dispatch(tmp_path, cfg, dc, base, measured, metadata, task)
    assert result['status'] == 'not_started_prerequisite'
    suite.atomic_json(tmp_path/'cells/v3_S1_gate/result.json', dict(status='completed', passed=False))
    assert suite.v3_dispatch(tmp_path, cfg, dc, base, measured, metadata, task)['status'] == 'not_started_prerequisite'


def test_v3_queue_and_whole_stage_budget_are_prospective():
    cfg, *_ = v3_config()
    tasks = suite.make_tasks(cfg, inventory(cfg))
    index = {t['id']: t for t in tasks}
    assert len(index) == len(tasks) == 7481
    assert sum(t['planned_solves'] for t in tasks) == 47808
    for task in tasks:
        assert all((index[d]['stage'], index[d]['layer']) < (task['stage'], task['layer']) for d in task['dependencies'])
    pilots = [dict(solver=s, elapsed_seconds=2.) for s in cfg['synthetic']['solvers']]
    estimate = suite.v3_stage_estimate(cfg, [t for t in tasks if t['stage'] == 1], pilots, 16)
    assert estimate['planned_solves'] == 3600
    assert estimate['estimated_seconds'] == pytest.approx(675.)


def test_v3_own_baseline_failure_keeps_selection_undefined(tmp_path):
    cfg, *_ = v3_config()
    ids = suite.v3_inventory(cfg, 'fixture')
    suite.atomic_json(tmp_path/'prepared/fixture.json', dict(trials=ids))
    panel = suite.v3_candidates(cfg, 'gain')
    cells = {}
    for candidate in panel:
        cells[candidate['id']] = []
        for inner, split in enumerate(suite.folds(ids, 0)['inner']):
            key = candidate['id']+str(inner)
            cells[candidate['id']].append(key)
            rows = [dict(ids[i], mode=mode, status='completed',
                center_metrics={m: dict(nmse=1.) for m in suite.MODALITIES})
                for i in split['validation'] for mode in ('center_EEG', 'center_fNIRS')]
            if candidate['id'] == 'baseline' and inner == 0:
                rows[0]['status'] = 'failed_domain'
            suite.atomic_json(tmp_path/f'cells/{key}/result.json', dict(status='completed', rows=rows))
    result = suite.v3_select(tmp_path, cfg, dict(subject='fixture', outer=0, candidate_panel=panel, inner_cells=cells))
    assert result['status'] == 'failed_contract' and 'selected' not in result
    assert result['candidates']['baseline']['completed_fits'] == 35


def test_v3_null_failure_preserves_joint_status_with_fixed_denominator(tmp_path):
    cfg, *_ = v3_config()
    task = dict(id='one', family='S2', kind='v3_fits', stage=2, layer=4, solver='O2', role='fixed',
        trials=[0], subject='subject_01', outer=0, dependencies=[], queue_position=0, planned_solves=14, timeout_seconds=900)
    suite.persist_tasks(tmp_path, [task])
    rows = [dict(mode=mode, status='completed', subject='subject_01', session='session_01', sample_id='one',
        center_metrics={m: dict(nmse=1.) for m in suite.MODALITIES}) for mode in suite.OUTER_MASKS]
    rows[-1]['status'] = 'failed_domain'
    suite.atomic_json(tmp_path/'cells/one/result.json', dict(status='completed', rows=rows, actual_solves=14))
    summary = suite.v3_summarize(tmp_path, cfg)
    assert summary['measured']['S2/O2']['full_panel_B'] is None
    assert summary['measured']['S2/O2']['all_14_modes_complete'] == 0
    import csv
    counts = list(csv.DictReader((tmp_path/'mode_status.csv').open()))
    joint = next(r for r in counts if r['rule'] == 'S2/O2' and r['mode'] == 'center_fNIRS')
    assert joint['completed'] == '1' and joint['expected'] == '72'


def test_v3_feature_checks_and_native_clock_noise_support():
    cfg, dc, base, *_ = v3_config()
    checks = suite.v3_engineering_checks(cfg, dc, base)
    assert all(c['passed'] for c in checks.values())
    assert checks['feature_masks_and_noise']['retained_ranks']['all_missing'] == 0
    assert checks['native_boundary']['existing_output_bit_identical']


def test_v3_report_missing_support_distinguishes_center_whole_and_unmasked_targets():
    from experiments.scripts import render_ssm_overnight_report as renderer
    truth = np.tile(np.sin(np.arange(120)[:, None]/8), (1, 4))
    estimate = truth+.05
    observed = np.ones((120, 3), dtype=bool)
    observed[52:68, 1:] = False
    result = renderer.v3_hidden_support_metrics(truth, estimate, observed, np.full((120, 4), .01))
    assert result['r']['hidden_samples'] == result['clean_HbR']['hidden_samples'] == 16
    assert result['clean_EEG']['hidden_samples'] == 0 and 'nrmse' not in result['clean_EEG']
    assert result['clean_HbR']['nrmse'] == pytest.approx(.05/np.std(truth[:, 3]))
    assert result['clean_HbR']['coverage95'] == 1.
    observed[:] = True
    observed[:, 0] = False
    result = renderer.v3_hidden_support_metrics(truth, estimate, observed)
    assert result['r']['hidden_samples'] == result['clean_EEG']['hidden_samples'] == 120
    assert result['clean_HbR']['hidden_samples'] == 0
    assert 'coverage95' not in result['r']


def test_v3_report_distinguishes_physical_failure_from_an_unused_start_budget():
    from experiments.scripts import render_ssm_overnight_report as renderer
    row = dict(status='failed_domain', failure_stage='MAP_path_physical_check',
               starts=[dict(convergence_reason='evaluation_budget')])
    assert renderer.v3_failure_reason({}, {}, row, {}) == 'MAP_path_physical_check'
    assert renderer.v3_failure_reason(dict(dependencies=['projection']), dict(status='data_unavailable'), None,
        {'projection': dict(status='failed_contract')}) == 'dependency_failure'


def test_v3_adaptation_missing_truth_keeps_the_paired_panel_incomplete(tmp_path):
    cfg, *_ = v3_config()
    tasks = []
    for panel in range(4):
        for trial in range(6):
            baseline, selected = f'b{panel}_{trial}', f's{panel}_{trial}'
            truth = {k: dict(nrmse=.5) for k in ('r','clean_EEG','clean_HbO','clean_HbR')}
            row = dict(status='completed', mode='full', truth=truth)
            suite.atomic_json(tmp_path/f'cells/{baseline}/result.json', dict(status='completed', rows=[row]))
            if panel == trial == 0:
                row = copy.deepcopy(row)
                row['truth']['r']['nrmse'] = None
            suite.atomic_json(tmp_path/f'cells/{selected}/result.json', dict(status='completed', rows=[row]))
            tasks.append(dict(id=selected, reference_task=baseline, family='S3_synthetic', role='selected',
                solver='O2', condition=dict(id='matched'), panel=panel))
    result = suite.v3_adaptation_screen(tmp_path, cfg, tasks, 'gain')
    matched = next(r for r in result['conditions'] if r['condition'] == 'matched')
    assert matched['paired_assessment_counts'] == [5,6,6,6]
    assert matched['verdict'] == 'incomplete' and not result['passed']


def test_model_drift_replay_closes_with_neural_process_noise_and_zero_hemodynamic_noise():
    _, _, base, *_ = v3_config()
    p, c, _ = suite.model(base, suite.BASE)
    rng = np.random.default_rng(6819)
    z = np.zeros((64, 6))
    z[0, 0] = .02
    for t in range(1, len(z)):
        z[t] = suite.core.rk4_transition(z[t-1], p, c)
        z[t, 0] += rng.normal()*.01
    replay, _, _ = suite.driver_replay(z, p, c, driver_law='model_drift')
    legacy, _, _ = suite.driver_replay(z, p, c)
    np.testing.assert_allclose(replay, z, atol=1e-5, rtol=0)
    assert np.max(abs(legacy-z)) > 1e-5


def test_v3_continuation_preserves_scientific_failures_and_original_budget(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    cfg, *_ = v3_config()
    monkeypatch.setattr(suite, 'ROOT', tmp_path)
    monkeypatch.setattr(suite, 'v3_summarize', lambda *args, **kwargs: None)
    root = tmp_path/cfg['output_root']; source = root/'stopped'; target = root/'continued'
    source.mkdir(parents=True)
    (source/'controller.lock').touch()
    started = (datetime.now(timezone.utc)-timedelta(minutes=3)).isoformat()
    reason = 'controller_exception: report fixture'
    suite.atomic_json(source/'manifest.json', dict(execution='stopped', stop_reason=reason,
        budget_started_at=started, schema=cfg['schema']))
    (source/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    for name in ('scope_inventory','fold_inventory','preflight','pilots','source_snapshot_identity','frozen_input_identity'):
        suite.atomic_json(source/(name+'.json'), {})
    tasks=[]
    for i, status in enumerate(('completed','failed_contract','not_started_budget')):
        task=dict(id=str(i), family='S1', kind='v3_temporal', layer=i, queue_position=i, planned_solves=1, timeout_seconds=900)
        tasks.append(task)
        row=dict(task_id=str(i), family='S1', kind='v3_temporal', status=status, rows=[],
                 reason=reason if i==2 else 'retained scientific outcome')
        suite.atomic_json(source/f'cells/{i}/result.json', row)
        suite.append_status(source, row)
    suite.persist_tasks(source, tasks)
    original=(source/'cells/1/result.json').read_bytes()
    suite.v3_prepare_continuation(target, source, cfg)
    assert (target/'cells/1/result.json').read_bytes()==original
    assert (source/'cells/1/result.json').read_bytes()==original
    assert not (target/'cells/2/result.json').exists()
    assert suite.read_json(target/'manifest.json')['budget_started_at']==started
    assert suite.read_json(source/'manifest.json')['execution']=='stopped'
    manifest=suite.read_json(source/'manifest.json')
    manifest['budget_started_at']=(datetime.now(timezone.utc)-timedelta(hours=9)).isoformat()
    suite.atomic_json(source/'manifest.json',manifest)
    with pytest.raises(ValueError,match='time budget is exhausted'):
        suite.v3_prepare_continuation(root/'expired',source,cfg)
    assert not (root/'expired').exists()


def test_v3_summary_accepts_inner_rows_without_truth_condition(tmp_path):
    cfg, *_ = v3_config()
    task=dict(id='inner',family='S3_synthetic',kind='v3_fits',stage=3,layer=8,solver='O2',role='inner',
        trials=[1],subject='fixture',outer=0,dependencies=[],queue_position=0,planned_solves=1,timeout_seconds=900)
    suite.persist_tasks(tmp_path,[task])
    suite.atomic_json(tmp_path/'cells/inner/result.json',dict(status='completed',rows=[dict(
        row_id='1__center_EEG',mode='center_EEG',status='completed',solver='O2')]))
    result=suite.v3_summarize(tmp_path,cfg)
    assert result['families']['S3_synthetic']['expected_cells']==1


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


def _blocking_budget_worker(*args):
    """Picklable spawn target: cannot finish inside the controller's budget."""
    time.sleep(10.)
    raise AssertionError('controller did not terminate the blocked test worker')


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
    monkeypatch.setattr(suite, 'execute_cell', _blocking_budget_worker)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        suite.run_scheduler(tmp_path, cfg)
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)
    result = suite.read_json(tmp_path/'manifest.json')
    assert result['execution'] == 'stopped_budget'
    assert result['stop_reason'] == 'hard_time_budget'
    assert suite.cell_result(tmp_path, 'pending')['status'] in ('timeout', 'not_started_budget')
    assert (tmp_path/'OVERNIGHT_REPORT.md').exists()
    assert 'pending' in (tmp_path/'failure_attribution.csv').read_text()
