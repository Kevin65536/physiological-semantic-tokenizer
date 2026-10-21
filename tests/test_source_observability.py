"""Synthetic source-information contracts; no retained or raw evidence needed."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.inference.shared_driver_modes import temporal_mode_basis
from src.inference.shared_driver_spatial import forward_spatial_shared_driver
from src.inference.source_observability import (
    fit_free_physiology, information_decomposition, nuisance_profile_information,
    pack_spatial_linearization, weighted_jacobian,
)
from src.inference.t3a_balloon_robust_ssm import BalloonFreeParameters, BalloonParameters


def fixture(t=48):
    basis = temporal_mode_basis(t, 4)
    coefficients = np.array([[.007], [.018], [-.011], [.008]])
    initial = np.array([[.008, .005, -.003, .002, -.001]])
    parameters = BalloonParameters(free=BalloonFreeParameters(tau=1.3, kappa=.45))
    kwargs = dict(temporal_basis=basis, eeg_loadings=np.array([[1.], [-.7]]),
                  hb_loadings=np.ones((1, 1)), numerical_backend='numba', substeps=4)
    truth = forward_spatial_shared_driver(coefficients, initial, parameters, .25,
        derivative=False, **kwargs)
    return coefficients, initial, parameters, kwargs, truth


def test_weighting_uses_fixed_scales_and_masks_without_renormalizing():
    jac = np.arange(24.).reshape(3, 2, 4)
    mask = np.array([[True, True], [False, True], [True, False]])
    poisoned = jac.copy()
    poisoned[~mask] = np.nan
    actual = weighted_jacobian(poisoned, [2., 4.], mask, [1., .5])
    expected = (jac*np.array([.5, .125])[None, :, None])[mask]
    np.testing.assert_array_equal(actual, expected)
    flat = weighted_jacobian(poisoned.reshape(6, 4), [2., 4.], mask, [1., .5])
    np.testing.assert_array_equal(flat, actual)
    full = weighted_jacobian(jac, [2., 4.], feature_weights=[1., .5])
    np.testing.assert_array_equal(full[mask.ravel()], actual)
    with pytest.raises(ValueError, match='boolean'):
        weighted_jacobian(jac, [1., 1.], np.ones((3, 2)))


def test_prior_does_not_inflate_data_rank_and_fixed_scales_remain_explicit():
    result = information_decomposition(np.diag([2., 0.]), np.diag([0., 3.]), [4., 5.])
    assert result['data_rank'] == 1
    assert result['total_rank'] == 2
    assert np.isinf(result['data_condition'])
    np.testing.assert_allclose(result['data_singular_values'], [8., 0.])
    np.testing.assert_allclose(result['prior_fraction_by_total_direction'], [1., 0.])
    np.testing.assert_allclose(result['parameter_scales'], [4., 5.])


def test_nuisance_compensation_projects_exact_alias_but_preserves_independent_info():
    # Last column has one perfectly compensable and one independent component.
    jac = np.array([[1., 2.], [0., 3.], [0., 0.]])
    result = nuisance_profile_information(jac, [1])
    np.testing.assert_allclose(result['compensating_nuisance'], [[-2.]])
    np.testing.assert_allclose(result['profiled_information'], [[9.]])
    assert result['compensation_fraction'][0] == pytest.approx(4/13)
    alias = nuisance_profile_information(np.array([[1., 2.], [2., 4.]]), [1])
    assert alias['compensation_fraction'][0] == pytest.approx(1.)
    assert alias['profiled_information'][0, 0] < 1e-25


def test_nested_observation_rows_cannot_reduce_data_information():
    rng = np.random.default_rng(4)
    jac = rng.normal(size=(6, 3, 5))
    small = np.zeros((6, 3), bool)
    small[:, :1] = True
    a = weighted_jacobian(jac, [2., 3., 4.], small)
    b = weighted_jacobian(jac, [2., 3., 4.])
    assert np.linalg.eigvalsh(b.T@b-a.T@a)[0] > -1e-12


def test_log_physiology_columns_and_separate_prior_derivatives_match_independent_difference():
    coefficient, initial, parameters, kwargs, truth = fixture(t=16)
    local = pack_spatial_linearization(coefficient, initial, parameters, .25,
        sd=np.ones(4), **kwargs)
    for column, parameter in enumerate(('tau', 'kappa')):
        prediction = []
        for sign in (-1, 1):
            changed = replace(parameters, free=replace(parameters.free,
                **{parameter: getattr(parameters.free, parameter)*np.exp(sign*2e-5)}))
            prediction.append(forward_spatial_shared_driver(coefficient, initial, changed,
                .25, derivative=False, **kwargs)['prediction'].ravel())
        numerical = (prediction[1]-prediction[0])/4e-5
        np.testing.assert_allclose(local['observation_jacobian'][:, -2+column], numerical, rtol=5e-6, atol=1e-10)
    assert local['parameter_names'][-2:] == ['log_tau', 'log_kappa']
    assert set(local['prior_residuals']) == {'driver_curvature', 'initial_state', 'flow_prior', 'physiology_prior'}
    np.testing.assert_allclose(local['prior_jacobian'], np.vstack(list(local['prior_jacobians'].values())))


def test_actual_free_physiology_oracle_recovers_nontruth_start_and_nonrest_initials():
    coefficient, initial, parameters, kwargs, truth = fixture(t=64)
    fit = fit_free_physiology(truth['prediction'], parameters, .25,
        sd=[.025, .025, .01, .005], start_tau_kappa=(2.3, .8),
        oracle_coefficients=coefficient, oracle_initial_coordinates=initial,
        driver_curvature_weight=0., initial_weight=0., flow_prior_weight=0.,
        max_nfev=80, gradient_tolerance=1e-8, **kwargs)
    assert fit['success'], fit['convergence_reason']
    assert fit['tau'] == pytest.approx(parameters.free.tau, rel=2e-5)
    assert fit['kappa'] == pytest.approx(parameters.free.kappa, rel=2e-5)
    assert fit['nfev'] > 1
    assert fit['objective_components']['data'] < 1e-14


def test_profile_refits_other_physiology_driver_and_initial_not_fixed_forward_sweep():
    coefficient, initial, parameters, kwargs, truth = fixture(t=48)
    fit = fit_free_physiology(truth['prediction'], parameters, .25,
        sd=[.025, .025, .01, .005], start_tau_kappa=(2.2, .9), fixed_tau=1.5,
        driver_curvature_weight=0., initial_weight=0., flow_prior_weight=0.,
        max_nfev=100, gradient_tolerance=1e-6, **kwargs)
    assert fit['tau'] == pytest.approx(1.5)
    assert fit['kappa'] != pytest.approx(.9)
    assert np.linalg.norm(fit['initial_coordinates']) > 1e-4
    assert fit['free_coordinates'][:-2].all()
    assert not fit['free_coordinates'][-2] and fit['free_coordinates'][-1]
    assert len(fit['trace']) > 1
    np.testing.assert_allclose(fit['objective'], sum(fit['objective_components'].values()))


def test_hidden_targets_never_affect_free_fit_or_warm_start():
    coefficient, initial, parameters, kwargs, truth = fixture(t=20)
    mask = np.ones(truth['prediction'].shape, bool)
    mask[:, :2] = False
    options = dict(sd=np.ones(4), visible=mask, max_nfev=5, **kwargs)
    clean = fit_free_physiology(truth['prediction'], parameters, .25, **options)
    for poison in (np.nan, np.inf, 1e30):
        target = truth['prediction'].copy()
        target[~mask] = poison
        actual = fit_free_physiology(target, parameters, .25, **options)
        np.testing.assert_array_equal(actual['prediction'], clean['prediction'])
        assert actual['objective'] == clean['objective']


def test_runner_contract_rejects_raw_or_other_retained_roots(tmp_path):
    path = Path(__file__).parents[1]/'experiments/scripts/evaluate_source_observability.py'
    spec = importlib.util.spec_from_file_location('observability_runner_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = runner.read_config()
    assert runner.check(cfg)['synthetic_shape'] == [120, 5]
    cfg['data']['spatial_run'] = '/tmp/unapproved'
    import yaml
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='boundary'):
        runner.read_config(config)


def test_summary_counts_failed_missing_metric_tasks_and_preserves_success_medians(tmp_path, monkeypatch):
    path = Path(__file__).parents[1]/'experiments/scripts/evaluate_source_observability.py'
    spec = importlib.util.spec_from_file_location('observability_summary_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = runner.read_config()
    tasks = [dict(id=str(i), kind='recovery', modality='Hb_only', prior_multiplier=0.,
                  noise_multiplier=1., seed=i, tau_truth=1.3, kappa_truth=.45) for i in range(3)]
    manifest = dict(tasks=tasks, planned=3, task_counts={'recovery': 3})
    out = tmp_path/'run'
    runner.write_json(out/'manifest.json', manifest)
    metrics = {name: .25 for name in ('theta_log_rmse', 'tau_absolute_error', 'kappa_absolute_error',
        'driver_nrmse', 'initial_physical_rmse', 'hb_clean_rmse', 'hb_clean_standardized_nrmse')}
    runner.write_json(out/'tasks/0.json', dict(**tasks[0], success=True, status='completed',
        metrics=metrics, objective=.5, objective_components={'data': .5}, elapsed_s=.1))
    runner.write_json(out/'tasks/1.json', dict(**tasks[1], success=False, status='failed_exception', elapsed_s=.1))
    support = tmp_path/runner.SUPPORT
    support.mkdir(parents=True)
    (support/'local_information.csv').write_text('dataset,gain\nsynthetic,1\n')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'figures', lambda *args: [])
    summary = runner.summarize(cfg, out)
    group = summary['groups']['recovery|Hb_only|prior=0|noise=1']
    assert (group['planned'], group['terminal'], group['succeeded'], group['failures'], group['pending']) == (3, 2, 1, 1, 1)
    assert group['medians']['theta_log_rmse'] == .25
