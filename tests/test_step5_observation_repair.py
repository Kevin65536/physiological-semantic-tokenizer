import copy
from dataclasses import replace

import numpy as np
import pytest

from src.inference.observation_baselines import (
    bridge_transform,
)
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal

from experiments import evaluate_step5_observation_repair as repair
from tests.test_step5_observation_diagnostic import synthetic_trials


def test_scale_bridge_closes_with_negative_control_retained():
    cfg, dc, base, _, _ = repair.load_config()
    base['model']['steps'] = 32
    cfg['scale']['grid_points'] = 3
    dc['curve']['quadrature_order'] = 7
    result = repair.scale_job(cfg, dc, base, 'reused', 0)
    assert result['passed']
    assert set(result['log_likelihood']) == {'original', 'observation_only', 'synchronized'}
    original = result['metrics']['original']['boundary_minus_zero_ll']
    assert result['metrics']['synchronized']['boundary_minus_zero_ll'] == pytest.approx(original, abs=1e-9)
    assert abs(result['metrics']['observation_only']['boundary_minus_zero_ll']-original) > .01


def test_baseline_batch_matches_independent_dense_gaussian_conditioning():
    _, dc, base, _, _ = repair.load_config()
    count = 8; dc['bridge']['baseline_samples'] = 3
    p, c = repair.step5.localization.model(base, 'W', 0.)
    generated = repair.generate_linear_gaussian(base, count, 54)
    operator = repair.trajectory_operator(count, 'baseline', dc)
    y = operator.apply(generated['observations'])
    variance = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
    result = repair.joint.smooth_balloon_trajectory_reference(y, p, config=c,
        trajectory_spec=operator, noise_variance=variance)
    # Independent block covariance construction from independent process-noise increments, then condition
    # on orthonormal output support. No pseudo-inverse from the implementation.
    _, a = repair.core.rk4_transition_with_jacobian(np.zeros(6), p, c)
    factor = np.zeros((6*count, 6*count))
    for t in range(count):
        factor[t*6:(t+1)*6, :6] = np.linalg.matrix_power(a, t)@np.diag(c.initial_state_std)
        for j in range(1, t+1):
            factor[t*6:(t+1)*6, j*6:(j+1)*6] = np.linalg.matrix_power(a, t-j)@np.diag(p.fixed.process_std)*np.sqrt(c.dt)
    prior = factor@factor.T
    baseline = np.eye(count)-np.ones((count, 1))@np.array([[1/3]*3+[0.]*(count-3)])
    matrix = np.kron(baseline, np.eye(3))
    h = np.kron(np.eye(count), repair.core.observation_jacobian(np.zeros(6), p))
    # y lies in null space of baseline averaging weights.
    weights = np.array([1/3]*3+[0.]*(count-3))
    basis = np.kron(null_space(weights[None, :]), np.eye(3))
    reduced = basis.T@matrix
    cross = prior@h.T@reduced.T
    predicted = reduced@(h@prior@h.T+np.diag(np.tile(variance, count)))@reduced.T
    target = basis.T@y.ravel()
    expected_mean = cross@np.linalg.solve(predicted, target)
    expected_cov = prior-cross@np.linalg.solve(predicted, cross.T)
    np.testing.assert_allclose(result['transformed_mean'].ravel(), expected_mean, atol=1e-11)
    np.testing.assert_allclose(result['transformed_covariance'], expected_cov, atol=1e-11)
    assert result['parameter_log_likelihood'] == pytest.approx(multivariate_normal.logpdf(target, cov=predicted), abs=1e-9)
    assert result['retained_rank'] == 3*(count-1)
    np.testing.assert_allclose(result['processed_noise_covariance'],
                               matrix@np.diag(np.tile(variance, count))@matrix.T, atol=1e-13)
    identity = repair.trajectory_operator(count, 'model', dc)
    full = repair.joint.smooth_balloon_trajectory_reference(generated['observations'], p, config=c,
        trajectory_spec=identity, noise_variance=variance)
    # Lost baseline information can only increase covariance under this model.
    assert np.linalg.eigvalsh(result['transformed_covariance']-full['transformed_covariance']).min() > -1e-12
    assert np.trace(result['transformed_covariance']) > np.trace(full['transformed_covariance'])


@pytest.mark.parametrize('variant', ['baseline', 'fnirs_filter', 'combined'])
def test_temporal_operator_matches_actual_pipeline_and_keeps_mixed_noise(variant):
    _, dc, _, _, _ = repair.load_config()
    rng = np.random.default_rng(95)
    values = rng.normal(size=(32, 3))
    operator = repair.trajectory_operator(32, variant, dc)
    np.testing.assert_allclose(operator.apply(values), bridge_transform(values, variant, dc), atol=2e-12)
    matrix, _ = operator.matrix()
    transformed_covariance = matrix@np.diag(np.tile([.3, .2, .1], 32))@matrix.T
    assert np.max(abs(transformed_covariance-np.diag(np.diag(transformed_covariance)))) > .001
    if variant == 'combined':
        assert [x['operation'] for x in operator.processing[:-1]] == [
            'resample_poly_roundtrip', 'bandpass_fnirs', 'subtract_weighted_baseline', 'known_common_scale']


def test_hidden_preprocessing_input_never_enters_temporal_likelihood():
    _, dc, base, _, _ = repair.load_config()
    p, c = repair.step5.localization.model(base, 'W', 0.)
    operator = repair.trajectory_operator(32, 'fnirs_filter', dc)
    values = np.random.default_rng(35).normal(size=(32, 3))*.01
    mask = np.ones_like(values, dtype=bool); mask[8, 1] = False
    malicious = values.copy(); malicious[8, 1] = 1e90
    a, b = operator.apply(values, mask), operator.apply(malicious, mask)
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[:, 1]).all()
    first = repair.joint.smooth_balloon_trajectory_reference(a, p, config=c, trajectory_spec=operator,
        noise_variance=[.01]*3, input_mask=mask)
    second = repair.joint.smooth_balloon_trajectory_reference(b, p, config=c, trajectory_spec=operator,
        noise_variance=[.01]*3, input_mask=mask)
    np.testing.assert_array_equal(first['transformed_mean'], second['transformed_mean'])
    empty = repair.joint.smooth_balloon_trajectory_reference(np.full((32, 3), np.nan), p, config=c,
        trajectory_spec=operator, noise_variance=[.01]*3, input_mask=np.zeros_like(mask))
    assert empty['retained_rank'] == 0 and empty['parameter_log_likelihood'] == 0
    np.testing.assert_array_equal(empty['transformed_mean'], np.zeros((32, 6)))


def test_batch_reference_rejects_impossible_baseline_support():
    _, dc, base, _, _ = repair.load_config()
    dc['bridge']['baseline_samples'] = 3
    p, c = repair.step5.localization.model(base, 'W', 0.)
    operator = repair.trajectory_operator(8, 'baseline', dc)
    with pytest.raises(ValueError, match='outside.*support'):
        repair.joint.smooth_balloon_trajectory_reference(np.ones((8, 3)), p, config=c,
            trajectory_spec=operator, noise_variance=[.01]*3)


def test_replay_boundary_rejects_out_of_scope_before_file_access(monkeypatch):
    cfg, dc, base, _, _ = repair.load_config()
    from pathlib import Path
    monkeypatch.setattr(Path, 'read_text', lambda *a, **k: pytest.fail('must reject before reading'))
    for subject in ('subject_02', 'subject_19', 'subject_24'):
        with pytest.raises(ValueError, match='boundary'):
            repair.load_replay_inputs(cfg, dc, base, subject)


def test_replay_reads_explicit_root_without_changing_module_root(tmp_path):
    import json

    cfg, dc, base, _, _ = repair.load_config()
    cfg['previous_run'] = 'retained'
    subject = dc['subjects'][0]
    identities = [dict(subject=subject, session=session, original_ma_trial_position=p)
                  for session in base['measured']['sessions'] for p in range(10)
                  if p not in base['measured']['heldout_trial_positions']]
    source_root = repair.ROOT
    for name, value in [('first', 1.), ('second', 2.)]:
        root = tmp_path/name
        previous = root/cfg['previous_run']
        previous.mkdir(parents=True)
        (previous/f'prepared_{subject}.json').write_text(json.dumps(dict(trials=identities)))
        np.savez(previous/f'prepared_{subject}.npz', **{
            coordinate: np.full((24, 120, 3), value) for coordinate in dc['coordinates']})
        arrays, detail = repair.load_replay_inputs(cfg, dc, base, subject, data_root=root)
        for array in arrays.values():
            np.testing.assert_array_equal(array, np.full((24, 120, 3), value))
        assert detail['trials'] == identities
        assert repair.ROOT == source_root


def test_replay_continues_after_a_trial_failure_and_marks_curve_incomplete(monkeypatch):
    cfg, dc, base, _, _ = repair.load_config()
    calls = []
    numerical = dict(minimum_evaluated_flow=1., saturation_evaluations=0, extreme_flow_evaluations=0)
    def fake_trace(y, *args):
        calls.append(y.copy())
        if len(calls) == 1:
            return dict(execution='failed', error='retained failure', numerical=numerical)
        return dict(execution='completed', numerical=numerical, trace=dict(increments=np.zeros(len(y))))
    monkeypatch.setattr(repair, 'traced_trial', fake_trace)
    result = repair.replay_curve_job(cfg, dc, base, np.zeros((3, 8, 3)), 'subject_01', 'broadband_pca', 'fNIRS_only', .25)
    assert len(calls) == 3 and all(np.isnan(y[:, 0]).all() for y in calls)
    assert result['completed_trials'] == 2 and result['curve_execution'] == 'failed'
    assert result['parameter_log_likelihood'] is None
    key = 'replay__subject_01__broadband_pca__fNIRS_only__04'
    old = dict(failures={f'curve__subject_01__{coordinate}__fNIRS_only__04': {}
                        for coordinate in ('broadband_pca', 'local_F3_alpha')})
    summary = repair.summarize(cfg, dc, dict(synthetic={}, replay={key: dict(execution='completed', result=result)}, paired={}), old)
    assert summary['replay']['restored_original_tasks'] == 0
    assert summary['replay']['unique_original_failed_tasks'] == 1
    assert summary['replay']['independent_new_trials'] == 0
    assert summary['replay']['curves']['subject_01__broadband_pca__fNIRS_only']['likelihood_maximum_w'] is None


def test_paired_cv_reuses_exact_linear_folds_and_training_noise_without_target_leakage(monkeypatch):
    from types import SimpleNamespace
    cfg, dc, base, measured, _ = repair.load_config()
    trials = synthetic_trials()
    calls = []
    def fake_smooth(y, parameters, **kwargs):
        calls.append((y.copy(), parameters.fixed.observation_scale))
        return SimpleNamespace(trajectory_mean=np.nan_to_num(y), total_observation_variance=np.ones_like(y), physical_checks={})
    monkeypatch.setattr(repair.joint, 'smooth_balloon_joint', fake_smooth)
    first = repair.paired_cv_job(cfg, dc, base, measured, trials, 'subject_01', 'EEG', 0, 1.)
    first_inputs = [y for y, _ in calls]
    assert all(np.isnan(y[52:68, 0]).all() for y in first_inputs)
    changed = copy.deepcopy(trials)
    for i in first['validation']:
        changed[i]['views']['target']['eeg_log_power'][52:68] += 1000
    calls.clear()
    second = repair.paired_cv_job(cfg, dc, base, measured, changed, 'subject_01', 'EEG', 0, 1.)
    assert first['observation_scale'] == second['observation_scale']
    assert first['selected_linear_alphas'] == second['selected_linear_alphas']
    for a, (b, _) in zip(first_inputs, calls):
        np.testing.assert_array_equal(a, b)
    assert first['rows'][0]['ssm']['joint']['score'] != second['rows'][0]['ssm']['joint']['score']


def test_flow_exit_precedes_rk4_and_matches_independent_physical_ode(monkeypatch):
    _, _, base, _, _ = repair.load_config()
    p, c = repair.step5.localization.model(base, 'W', 0.)
    z = np.array([-.22675076802403443, -.07572673509915534,
                  -5.741747298655799, -.7154405874587865, -.7219825036238462, -.048435160291565485])
    def rhs(t, u):
        r, s, f = u
        return [-.45*r, r-.64*s-.32*(f-1), s]
    def zero(t, u):
        return u[2]
    reference = solve_ivp(rhs, [0, c.dt], [z[0], z[1], np.exp(z[2])],
                          events=zero, method='DOP853', rtol=1e-12, atol=1e-14)
    diagnostic = repair.core.flow_drift_diagnostic(z, p, c.dt)
    assert diagnostic['first_zero_time_s'] == pytest.approx(reference.t_events[0][0], abs=1e-11)
    assert diagnostic['first_zero_time_s'] == pytest.approx(.0441994817463, abs=1e-11)
    monkeypatch.setattr(repair.core, '_rk4_step', lambda *a: pytest.fail('must classify before RK4'))
    monkeypatch.setattr(repair.core, '_rk4_step_with_jacobian', lambda *a: pytest.fail('must classify before RK4'))
    for transition in (repair.core.rk4_transition, repair.core.rk4_transition_with_jacobian):
        for substeps in (2, 8, 32):
            with pytest.raises(repair.core.FlowDomainExit, match='flow_domain_exit') as error:
                transition(z, p, replace(c, rk4_substeps=substeps))
            assert error.value.diagnostic == diagnostic


@pytest.mark.parametrize('kappa', [.1, 2., 3.])
def test_flow_minimum_includes_interior_extrema_in_all_damping_regimes(kappa):
    p = repair.core.BalloonParameters(
        fixed=repair.core.BalloonFixedParameters(gamma=1.),
        free=repair.core.BalloonFreeParameters(kappa=kappa))
    z = np.array([0., -2., np.log(.1), 0., 0., 0.])
    result = repair.core.flow_drift_diagnostic(z, p, 6.)
    solution = solve_ivp(lambda t, u: [-kappa*u[0]-(u[1]-1), u[0]],
        [0, 6], [-2., .1], dense_output=True, method='DOP853', rtol=1e-12, atol=1e-13)
    grid_minimum = solution.sol(np.linspace(0, 6, 10001))[1].min()
    assert result['minimum_flow'] == pytest.approx(grid_minimum, abs=2e-7)
    assert result['end_flow'] > 0 and not result['in_domain']
    assert 0 < result['first_zero_time_s'] < result['minimum_time_s'] < 6


def test_flow_failure_carries_exact_recent_observation_updates(monkeypatch):
    _, _, base, _, _ = repair.load_config()
    p, c = repair.step5.localization.model(base, 'W', 0.)
    count = 0
    def update(mean, covariance, *args):
        nonlocal count
        count += 1
        mean = mean.copy()
        if count == 5:
            mean[:3] = [-.22675, -.075727, np.log(.003209)]
        return mean, covariance, -.5, 0.
    monkeypatch.setattr(repair.joint, 'joint_observation_update', update)
    values = np.zeros((8, 3)); values[:, 0] = np.nan
    with pytest.raises(repair.core.FlowDomainExit) as error:
        repair.joint.parameter_log_likelihood(values, p, config=c)
    result = error.value.diagnostic
    assert result['transition_index'] == 5
    assert [u['time_index'] for u in result['update_history']] == [1, 2, 3, 4]
    for row in result['update_history']:
        assert row['observation_mask'] == [False, True, True]
        assert np.shape(row['predicted_covariance']) == (6, 6)
        assert np.shape(row['filtered_covariance']) == (6, 6)
    np.testing.assert_array_equal(result['update_history'][-1]['filtered_mean'],
                                  result['initial_transformed_state'])


@pytest.mark.parametrize('count', [64, 120])
@pytest.mark.parametrize('variant', ['fnirs_filter', 'combined'])
@pytest.mark.parametrize('mask_name', ['full', 'center_EEG', 'center_fNIRS', 'whole_EEG', 'whole_fNIRS'])
def test_mask_specific_operator_matches_visible_interpolation_pipeline(count, variant, mask_name):
    _, dc, _, _, _ = repair.load_config()
    values = np.random.default_rng(294).normal(size=(count, 3))
    mask = np.ones_like(values, bool)
    if mask_name != 'full':
        mask = np.isfinite(repair.step5.masked_input(values, mask_name)[0])
    base = repair.trajectory_operator(count, variant, dc)
    operator = base.with_visible_interpolation(mask, mask)
    expected = np.zeros_like(values)
    for channel in range(3):
        visible = np.flatnonzero(mask[:, channel])
        if len(visible):
            expected[:, channel] = np.interp(np.arange(count), visible, values[visible, channel])
    expected = bridge_transform(expected, variant, dc)
    expected[~mask] = np.nan
    changed = values.copy(); changed[~mask] = np.inf
    np.testing.assert_allclose(operator.apply(values, mask), expected, atol=3e-12)
    np.testing.assert_array_equal(operator.apply(values, mask), operator.apply(changed, mask))
    matrix, available = operator.matrix(mask)
    np.testing.assert_array_equal(available, mask)
    assert np.all(matrix[:, ~mask.ravel()] == 0)
    assert np.all(matrix[~mask.ravel()] == 0)
    if mask_name == 'center_fNIRS':
        assert base.matrix(mask)[1][:, 1:].sum() == 0
        assert available[:, 1:].sum() == 2*(count-16)


def test_mask_specific_gaussian_mean_and_full_noise_use_same_operator():
    _, dc, base, _, _ = repair.load_config()
    dc['bridge']['baseline_samples'] = 3
    p, c = repair.step5.localization.model(base, 'W', 0.)
    values = repair.generate_linear_gaussian(base, 8, 194)['observations']
    mask = np.ones_like(values, bool); mask[4:6, 1:] = False
    operator = repair.trajectory_operator(8, 'baseline', dc).with_visible_interpolation(mask, mask)
    noise = np.array([.02, .03, .04])
    result = repair.joint.smooth_balloon_trajectory_reference(operator.apply(values, mask), p,
        config=c, trajectory_spec=operator, noise_variance=noise, input_mask=mask)
    matrix, available = operator.matrix(mask)
    selected = matrix[available.ravel()]
    h = np.kron(np.eye(8), repair.core.observation_jacobian(np.zeros(6), p))
    prior = repair.joint.linearized_trajectory_prior(8, p, c)
    cross = prior @ h.T @ selected.T
    covariance = selected @ (h@prior@h.T+np.diag(np.tile(noise, 8))) @ selected.T
    inverse = np.linalg.pinv(covariance, rcond=1e-12)
    mean = cross @ inverse @ operator.apply(values, mask)[available]
    np.testing.assert_allclose(result['transformed_mean'].ravel(), mean, atol=1e-11)
    np.testing.assert_allclose(result['transformed_covariance'], prior-cross@inverse@cross.T, atol=1e-11)
    np.testing.assert_allclose(result['processed_noise_covariance'],
                               (matrix*np.tile(noise, 8))@matrix.T, atol=1e-13)
    assert result['observed_coordinates'] == mask.sum()
    assert result['retained_rank'] < mask.sum()  # baseline loss, never fake independent filled points
    changed = values.copy(); changed[~mask] = 1e90
    other = repair.joint.smooth_balloon_trajectory_reference(operator.apply(changed, mask), p,
        config=c, trajectory_spec=operator, noise_variance=noise, input_mask=mask)
    np.testing.assert_array_equal(result['transformed_mean'], other['transformed_mean'])


def test_interpolation_rejects_insufficient_context_and_handles_irregular_clocks():
    clock = np.array([0., .2, .9, 1.5])
    operator = repair.joint.TrajectoryObservationSpec(np.array([np.eye(4)]*3), clock, clock, ())
    mask = np.zeros((4, 3), bool); mask[[0, 3], 0] = True
    mapped = operator.with_visible_interpolation(mask, np.ones_like(mask))
    values = np.zeros((4, 3)); values[3, 0] = 3.
    np.testing.assert_allclose(mapped.apply(values, mask)[:, 0], 2*clock)
    assert np.isnan(mapped.apply(values, mask)[:, 1:]).all()
    mask[3, 0] = False
    with pytest.raises(ValueError, match='insufficient'):
        operator.with_visible_interpolation(mask, mask)


def test_v2_rejects_scope_expansion_and_measured_entry_before_array_access(monkeypatch, tmp_path):
    from types import SimpleNamespace
    cfg, dc, base, measured, metadata = repair.load_config(repair.MASK_CONFIG)
    monkeypatch.setattr(repair, 'load_replay_inputs', lambda *a: pytest.fail('no data access'))
    for trial in [['subject_24', 4], ['subject_01', 5]]:
        changed = copy.deepcopy(cfg); changed['flow']['trials'][0] = trial
        with pytest.raises(ValueError, match='boundary'):
            repair.load_mask_repair_config(changed)
    with pytest.raises(ValueError, match='nonlinear Student-t temporal'):
        repair.run_mask_repair(SimpleNamespace(stage='measured', run_dir=tmp_path),
                               cfg, dc, base, measured, metadata)


def test_v2_synthetic_case_exercises_all_masks_and_retains_inference_failures(monkeypatch):
    cfg, dc, base, _, _ = repair.load_config(repair.MASK_CONFIG)
    # Tiny independent fixture, with the same five observation tasks.
    cfg['trajectory']['steps'] = 24
    cfg['trajectory']['center_steps'] = 4
    cfg['trajectory']['variants'] = ['model']
    def failed(*args, **kwargs):
        raise FloatingPointError('deliberate pointwise fixture failure')
    monkeypatch.setattr(repair.joint, 'smooth_balloon_joint', failed)
    result = repair.mask_bridge_job(cfg, dc, base, 'linearized_gaussian', 0)
    assert len(result['rows']) == 20
    assert sum(row['execution'] == 'failed' for row in result['rows']) == 10
    assert all(row['execution'] == 'completed' for row in result['rows']
               if row['branch'] == 'mask_specific_gaussian_reference')
    for row in result['operator_checks'].values():
        assert row['visible_inputs'] == row['visible_outputs']
        assert row['hidden_input_intervention_max_error'] == 0.
    summary = repair.summarize_mask_repair(cfg, {'fixture': dict(execution='completed', result=result)}, {}, [])
    invariant = summary['groups']['linearized_gaussian__model__whole_fNIRS__mask_specific_gaussian_reference']
    assert invariant['w_interpretation'] == 'EEG_only_no_W_information'
    assert invariant['slow_preference_count'] is None
    assert invariant['mean_boundary_minus_zero_ll'] is None
    assert invariant['maximum_absolute_w_ll_difference'] < 1e-7
