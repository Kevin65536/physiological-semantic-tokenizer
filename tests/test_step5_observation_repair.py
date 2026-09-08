import copy

import numpy as np
import pytest
from scipy.linalg import null_space
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
    # Independent block covariance construction from innovations, then condition
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
    np.testing.assert_allclose(operator.apply(values), repair.diagnostic.bridge_transform(values, variant, dc), atol=2e-12)
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
