"""Synthetic contracts for the four-state Tak mismatch numerical owner."""
import numpy as np
import pytest

from src.inference import tak_mismatch as tm
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import BalloonParameters, BalloonFreeParameters


def parameters():
    return BalloonParameters(free=BalloonFreeParameters(kappa=.64, tau=2.))


def reference_operator(n):
    reference = np.eye(n)
    reference[:, :3] -= 1/3
    return np.kron(reference, np.eye(3))


def test_kernel_endpoints_zero_integral_and_analytic_derivative():
    t = np.linspace(-5, 5, 10001)
    phi, derivative = tm.weak_balance_kernel(t, 10)
    np.testing.assert_array_equal(phi[[0, -1]], [0, 0])
    np.testing.assert_array_equal(derivative[[0, -1]], [0, 0])
    np.testing.assert_allclose(np.trapezoid(phi, t), 0, atol=1e-15)
    np.testing.assert_allclose(np.trapezoid(derivative, t), 0, atol=1e-13)
    eps = 1e-5
    p1, _ = tm.weak_balance_kernel(t[1:-1]+eps, 10)
    p0, _ = tm.weak_balance_kernel(t[1:-1]-eps, 10)
    np.testing.assert_allclose((p1-p0)/(2*eps), derivative[1:-1], atol=3e-11)


def test_weak_balance_linear_solution_reference_and_scale_invariance():
    p = parameters()
    dt = .05
    time = np.arange(1801)*dt
    omega = .35
    eta = 1+(1-p.fixed.E0)*np.log1p(-p.fixed.E0)/p.fixed.E0
    a = eta*(p.free.tau+2)-2
    b = 1+(eta-1)/p.fixed.alpha
    rho = p.fixed.Q0/p.fixed.P0
    sin_coeff = rho*(b+p.free.tau*a*omega**2)/(1+(p.free.tau*omega)**2)
    cos_coeff = rho*omega*(a-p.free.tau*b)/(1+(p.free.tau*omega)**2)
    total = np.sin(omega*time)
    deoxy = sin_coeff*np.sin(omega*time)+cos_coeff*np.cos(omega*time)
    hb = np.column_stack((total-deoxy, deoxy))
    baseline = tm.weak_balance_residual(hb, dt, p)
    shifted = tm.weak_balance_residual(hb+[2., -7.], dt, p)
    scaled = tm.weak_balance_residual(13*hb, dt, p)
    for width, result in baseline['scales'].items():
        assert result['status'] == 'completed'
        assert result['D_bal'] < 5e-5
        np.testing.assert_allclose(np.diff(result['center_time_s']), 1)
        assert result['center_time_s'][0] >= width/2
        assert result['center_time_s'][-1] <= time[-1]-width/2
        np.testing.assert_allclose(result['w'], shifted['scales'][width]['w'], atol=3e-14)
        np.testing.assert_allclose(result['D_bal'], scaled['scales'][width]['D_bal'], atol=1e-14)


def test_weak_balance_denominator_missing_full_support_and_unexcited_states():
    p = parameters()
    dt = .1
    t = np.arange(301)*dt
    hb = np.column_stack((np.sin(t*.7), .3*np.cos(t*.4)))
    valid = np.ones_like(hb, bool)
    valid[150, 0] = False
    hb[150, 0] = np.nan
    result = tm.weak_balance_residual(hb, dt, p, valid=valid)
    assert result['scales'][40.]['status'] == 'unavailable_complete_support'
    for width in (10., 20.):
        r = result['scales'][width]
        expected = abs(r['center_time_s']-15) > width/2
        np.testing.assert_array_equal(r['valid'], expected)
        assert np.isnan(r['w'][~expected]).all()
        if expected.any():
            manual = np.sqrt(np.nansum(r['w']**2)/(2*np.nansum(r['U']**2+r['V']**2)))
            assert r['D_bal'] == pytest.approx(manual)
            assert 0 <= r['D_bal'] <= 1
    quiet = tm.weak_balance_residual(np.ones((301, 2))*[4, -3], dt, p)
    r = quiet['scales'][10.]
    assert r['status'] == 'insufficient_dynamic_excitation'
    assert np.isnan(r['D_bal']) and r['denominator'] == 0
    absent = tm.weak_balance_residual(np.full((301, 2), np.nan), dt, p)
    assert absent['scales'][10.]['status'] == 'unavailable_data_support'
    with pytest.raises(ValueError, match='even integer'):
        tm.weak_balance_residual(np.zeros((301, 2)), dt, p, support_widths_s=(10.1,))


def test_training_sd_scores_common_support_components_and_original_units():
    y = np.zeros((5, 3))
    hp = np.broadcast_to([2., 6., 12.], y.shape).copy()
    jp = np.broadcast_to([4., 3., 8.], y.shape).copy()
    visible = np.ones_like(y, bool)
    visible[0, 1] = False
    y[0, 1:] = [np.nan, 10000]
    result = tm.score_tak_mismatch(y, hp, jp, [2, 3, 4], visible=visible)
    assert result['D1'] == pytest.approx(np.sqrt((2**2+3**2)/2))
    assert result['D_EH'] == 1
    assert result['D4'] == pytest.approx(np.sqrt((2**2+1**2+2**2)/3))
    assert [result['D_J_E'], result['D_J_O'], result['D_J_R']] == [2, 1, 2]
    assert result['hb_time_count'] == 4 and result['hb_observation_count'] == 8
    assert result['hb_only_components']['HbR']['rmse'] == 12
    np.testing.assert_array_equal(result['hb_only_components']['EEG']['standardized_residual'], -1)
    # A failed prediction on a declared scoring sample is not silently dropped.
    hp[2, 1] = np.nan
    failed = tm.score_tak_mismatch(y, hp, jp, [2, 3, 4], visible=visible)
    assert np.isnan(failed['D1'])
    assert failed['hb_only_components']['HbO']['status'] == 'failed_prediction'


def test_visibility_fits_are_independent_of_hidden_values_and_start_at_rest():
    p, n, dt = parameters(), 18, .25
    drive = .02*np.sin(np.arange(n)*.3)
    y = nonlinear_driver_forward(drive, [0, 1, 1, 1], p, dt)['canonical_prediction'][None]
    controls = dict(max_iterations=100, penalty=.002, gradient_tolerance=1e-5)
    all_views = tm.fit_tak_views(y, p, dt, [.03, .03, .03], **controls)
    assert all(view['converged'] for view in all_views.values())
    changed = y.copy()
    changed[..., 0] = 1e6
    hb_only = tm.fit_tak_views(changed, p, dt, [.03, .03, .03], views=('hb_only',), **controls)
    np.testing.assert_array_equal(hb_only['hb_only']['driver'], all_views['hb_only']['driver'])
    changed = y.copy()
    changed[..., 1:] = -1e6
    eeg_only = tm.fit_tak_views(changed, p, dt, [.03, .03, .03], views=('eeg_only',), **controls)
    np.testing.assert_array_equal(eeg_only['eeg_only']['driver'], all_views['eeg_only']['driver'])
    with pytest.raises(ValueError, match='starts'):
        tm.fit_tak_views(y, p, dt, [.03]*3, starts=[])


def test_fit_dispatch_sanitizes_hidden_data_and_shares_group_parameters(monkeypatch):
    calls = []
    def fake(y, p, dt, **kwargs):
        calls.append((y, kwargs))
        return {'status': 'completed', 'converged': True}
    monkeypatch.setattr(tm, 'fit_nonlinear_shared_parameters', fake)
    y = np.arange(48.).reshape(2, 8, 3)
    tm.fit_tak_views(y, parameters(), .1, [1, 2, 3], parameter_names=('kappa', 'tau'),
                     parameter_bounds=((.1, 2), (.5, 10)))
    assert len(calls) == 3
    for observation, control in calls:
        assert np.isnan(observation[~control['visible']]).all()
        np.testing.assert_array_equal(control['starts'][0]['driver'], np.zeros((2, 8)))
        np.testing.assert_array_equal(control['starts'][0]['initial_state'], [[0, 1, 1, 1]]*2)
        np.testing.assert_array_equal(control['starts'][0]['parameter_values'], [.64, 2])


def test_same_state_nonlinear_replay_gap_and_finer_integrator():
    p, n, dt = parameters(), 32, .25
    rh = .02*np.sin(np.arange(n)*.2)
    re = rh+.01*np.cos(np.arange(n)*.3)
    op = reference_operator(n)
    result = tm.nonlinear_drive_gap(re, rh, p, dt, [.1, .03, .02], mean_operator=op)
    assert result['status'] == 'completed' and result['D_drive_H'] > 0
    manual = np.sqrt(np.mean((result['difference']/[.03, .02])**2))
    assert result['D_drive_H'] == pytest.approx(manual)
    exact = tm.nonlinear_drive_gap(rh, rh, p, dt, [.03, .02], mean_operator=op)
    assert exact['D_drive_H'] == 0
    fine = tm.nonlinear_drive_gap(re, rh, p, dt, [.03, .02], mean_operator=op, substeps=8)
    assert abs(fine['D_drive_H']-result['D_drive_H']) < 1e-7


def observable_fixture(n=32):
    p, dt = parameters(), .25
    rh = .02*np.sin(np.arange(n)*.2)
    re = rh+.01*np.cos(np.arange(n)*.3)
    return (re, rh, rh, [0., 1., 1., 1.], p, dt, [.03, .02])


def test_projected_observable_gap_nuisance_dc_and_svd_normalization():
    args = observable_fixture()
    result = tm.observable_drive_gap(*args)
    assert result['status'] == 'uncalibrated_local_diagnostic'
    assert np.isnan(result['D_obs'])
    assert result['numerical_rank'] > 0
    projected = result['projected_driver_jacobian']
    expected = np.linalg.norm(projected@(args[0]-args[1]))/np.sqrt(result['hb_observation_count'])
    assert result['D_obs_allrank'] == pytest.approx(expected, abs=1e-13)
    np.testing.assert_allclose(projected@np.ones(len(args[0])), 0, atol=1e-12)
    shifted = tm.observable_drive_gap(args[0]+3, *args[1:])
    assert shifted['D_obs_allrank'] == pytest.approx(result['D_obs_allrank'], abs=1e-12)
    expanded = tm.observable_drive_gap(*args, nuisance_parameters=('kappa', 'tau'))
    assert expanded['nuisance_rank'] >= result['nuisance_rank']
    assert expanded['D_obs_allrank'] <= result['D_obs_allrank']+1e-12


def test_full_initial_span_cannot_promote_roundoff_to_observable_rank():
    args = observable_fixture()
    valid = np.zeros((len(args[0]), 2), bool)
    valid[2] = True
    result = tm.observable_drive_gap(*args, valid=valid)
    assert result['numerical_rank'] == 0
    assert result['status'] == 'insufficient_observable_rank'
    assert np.isnan(result['D_obs']) and np.isnan(result['D_obs_allrank'])


def test_domain_failures_remain_unavailable_scores():
    args = observable_fixture()
    invalid_initial = [0., 1., -1., 1.]
    observable = tm.observable_drive_gap(*args[:3], invalid_initial, *args[4:])
    assert observable['status'] == 'failed_physical_domain'
    assert np.isnan(observable['D_obs']) and np.isnan(observable['D_obs_allrank'])
    gap = tm.nonlinear_drive_gap(args[0], args[1], args[4], args[5], args[6],
                                initial_physical=invalid_initial)
    assert gap['status'] == 'failed_physical_domain' and np.isnan(gap['D_drive_H'])


def test_synthetic_noise_mode_calibration_reproducible_mask_and_no_fake_zero():
    args = observable_fixture()
    valid = np.ones((len(args[0]), 2), bool)
    valid[6:9] = False
    observable = tm.observable_drive_gap(*args, valid=valid)
    options = dict(driver_train_sd=.03, driver_steps=len(args[0]), noise_sd=.1,
                   ar1=.9, cross_correlation=.3, draws=128, seed=18)
    a = tm.calibrate_observable_modes(observable, **options)
    b = tm.calibrate_observable_modes(observable, **options)
    np.testing.assert_array_equal(a['projected_noise_threshold'], b['projected_noise_threshold'])
    manual = (options['driver_train_sd']*np.sqrt(options['driver_steps'])*observable['singular_values']
              > a['projected_noise_threshold'])
    manual[observable['numerical_rank']:] = False
    np.testing.assert_array_equal(a['retained_modes'], manual)
    assert a['interpretation'] == 'synthetic_noise_sensitivity_not_measured_identifiability'
    no_modes = tm.calibrate_observable_modes(observable, **dict(options, noise_sd=1e10))
    assert no_modes['retained_rank'] == 0
    assert no_modes['status'] == 'insufficient_observable_rank'
    assert np.isnan(no_modes['D_obs'])


def test_observable_scenarios_one_svd_two_projections_and_equivalent_calibration(monkeypatch):
    args = observable_fixture()
    valid = np.ones((len(args[0]), 2), bool)
    valid[8:11] = False
    calls = dict(svd=0, projection=0)
    original_observable = tm.observable_drive_gap
    original_quantile = tm._noise_mode_quantile
    def observable(*args, **kwargs):
        calls['svd'] += 1
        return original_observable(*args, **kwargs)
    def quantile(*args, **kwargs):
        calls['projection'] += 1
        return original_quantile(*args, **kwargs)
    monkeypatch.setattr(tm, 'observable_drive_gap', observable)
    monkeypatch.setattr(tm, '_noise_mode_quantile', quantile)
    options = dict(driver_train_sd=.5, draws=256, seed=18, quantile=.95)
    result = tm.observable_drive_scenarios(*args, valid=valid, **options)
    assert calls == dict(svd=1, projection=2)
    assert result['noise_projection_groups'] == 2
    assert result['primary_noise_scenario'] == 'primary_ar1_sd0p1'
    primary = result['scenarios'][result['primary_noise_scenario']]
    assert result['D_obs'] == primary['D_obs']
    assert primary['retained_rank'] > 0
    assert (primary['noise_sd'], primary['ar1'], primary['cross_correlation']) == (.1, .9, .3)
    for scenario in result['scenarios'].values():
        independent = tm.calibrate_observable_modes(result['observable'],
            driver_steps=len(args[0]), noise_sd=scenario['noise_sd'], ar1=scenario['ar1'],
            cross_correlation=scenario['cross_correlation'], **options)
        np.testing.assert_allclose(scenario['projected_noise_threshold'],
                                   independent['projected_noise_threshold'], rtol=2e-15)
        np.testing.assert_array_equal(scenario['retained_modes'], independent['retained_modes'])
        assert scenario['D_obs'] == pytest.approx(independent['D_obs'], nan_ok=True)
        curve = scenario['standardized_hb_difference']
        assert np.isnan(curve[~valid]).all()
        if scenario['retained_rank']:
            assert np.sqrt(np.mean(curve[valid]**2)) == pytest.approx(scenario['D_obs'])
    low = result['scenarios']['ar1_sd0p05']
    high = result['scenarios']['ar1_sd0p2']
    np.testing.assert_array_equal(low['projected_noise_threshold'], primary['projected_noise_threshold']/2)
    np.testing.assert_array_equal(high['projected_noise_threshold'], primary['projected_noise_threshold']*2)
    assert high['retained_rank'] <= primary['retained_rank'] <= low['retained_rank']


@pytest.mark.parametrize('failure', ['rank', 'domain', 'support'])
def test_observable_scenarios_failures_never_become_zero(failure):
    args = list(observable_fixture())
    valid = np.ones((len(args[0]), 2), bool)
    if failure == 'rank':
        valid[:] = False
        valid[2] = True
        status = 'insufficient_observable_rank'
    elif failure == 'domain':
        args[3] = [0., 1., -1., 1.]
        status = 'failed_physical_domain'
    else:
        valid[:] = False
        status = 'unavailable_data_support'
    result = tm.observable_drive_scenarios(*args, driver_train_sd=.03, valid=valid)
    assert result['status'] == status
    assert result['retained_rank'] == 0 and result['noise_projection_groups'] == 0
    assert np.isnan(result['D_obs']) and np.isnan(result['D_obs_allrank'])
    for scenario in result['scenarios'].values():
        assert scenario['status'] == status
        assert np.isnan(scenario['D_obs'])
        assert np.isnan(scenario['standardized_hb_difference']).all()


def test_observable_custom_scenario_primary_and_current_dof_nuisance():
    args = observable_fixture()
    scenarios = [dict(id='declared_high_noise', noise_sd=1e10, ar1=.9, cross_correlation=.3)]
    result = tm.observable_drive_scenarios(*args, driver_train_sd=.03,
        noise_scenarios=scenarios, nuisance_parameters=('kappa', 'tau'))
    assert result['primary_noise_scenario'] == 'declared_high_noise'
    assert result['observable']['nuisance_parameters'] == ('kappa', 'tau')
    assert result['status'] == 'insufficient_observable_rank'
    assert np.isnan(result['D_obs']) and np.isfinite(result['D_obs_allrank'])
    with pytest.raises(ValueError, match='ids'):
        tm.observable_drive_scenarios(*args, driver_train_sd=.03, noise_scenarios=scenarios*2)
    with pytest.raises(ValueError, match='thresholds'):
        tm.observable_drive_scenarios(*args, driver_train_sd=.03, singular_value_threshold=.1)


@pytest.mark.parametrize('ar1,cross_correlation', [(0., 0.), (.9, .3)])
def test_mode_noise_threshold_matches_analytic_stationary_covariance(ar1, cross_correlation):
    # A constant unit left vector has a closed-form variance under a stationary
    # two-chromophore AR(1) process, independent of the implementation's draws.
    n = 20
    observable = dict(hb_common_mask=np.ones((n, 2), bool),
        left_singular_vectors=np.ones((2*n, 1))/np.sqrt(2*n),
        singular_values=np.array([2.]), numerical_rank=1,
        driver_mode_coordinates=np.array([.2]))
    noise_sd = .1
    result = tm.calibrate_observable_modes(observable, driver_train_sd=.1,
        driver_steps=n, noise_sd=noise_sd, ar1=ar1,
        cross_correlation=cross_correlation, draws=8192, seed=300)
    time_covariance_sum = n+2*sum((n-lag)*ar1**lag for lag in range(1, n))
    variance = noise_sd**2*(1+cross_correlation)*time_covariance_sum/n
    expected_abs_q95 = 1.959963984540054*np.sqrt(variance)
    assert result['projected_noise_threshold'][0] == pytest.approx(expected_abs_q95, rel=.06)
