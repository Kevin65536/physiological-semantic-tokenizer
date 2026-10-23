"""Synthetic forward-only Hb subsystem checks, without measured-data access."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.inference import shared_driver_reconstruction as lib
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def cosine_basis(steps, modes):
    return np.cos(np.pi*(np.arange(steps)[:, None]+.5)*np.arange(modes)/steps)


def test_inlet_balance_independent_initial_p_minus_v_decays_by_exact_identity():
    parameters, steps, dt = BalloonParameters(), 30, .2
    log_flow = .14*np.sin(np.arange(steps)*.23)
    initial = [1.05, 1.16, .97]
    forward = lib.vascular_flow_forward(log_flow, initial, parameters, dt, substeps=8)
    time = np.arange(steps)*dt
    tau, alpha = parameters.free.tau, parameters.fixed.alpha

    def rhs(t, state):
        flow = np.exp(np.interp(t, time, log_flow))
        g = state[0]**(1/alpha-1)
        # Third coordinate is the integral in d(t)=d(0)*exp(-integral).
        return [(flow-state[0]*g)/tau, (flow-g*state[1])/tau, g/tau]

    independent = solve_ivp(rhs, (0, time[-1]), [initial[0], initial[1], 0.],
        t_eval=time, rtol=2e-11, atol=2e-13, max_step=dt/8).y.T
    expected = (initial[1]-initial[0])*np.exp(-independent[:, 2])
    np.testing.assert_allclose(forward['states'][:, 1]-forward['states'][:, 0],
                               expected, atol=2e-9, rtol=2e-7)
    assert np.all(np.diff(expected) < 0)
    assert forward['states'][0, 0] != forward['states'][0, 1]


@pytest.mark.parametrize('backend', ['python', 'numba'])
def test_discrete_sensitivities_include_interpolated_flow_and_three_initials(backend):
    parameters, steps = BalloonParameters(), 11
    basis = cosine_basis(steps, 4)
    coefficients = np.array([.04, -.08, .035, .012])
    initial = np.array([.98, 1.04, 1.01])
    options = dict(temporal_basis=basis, numerical_backend=backend)
    result = lib.vascular_flow_forward(coefficients, initial, parameters, .3, **options)
    vector, epsilon = np.r_[coefficients, np.log(initial)], 1e-6
    numeric = []
    for col in range(len(vector)):
        predictions = []
        for sign in (1, -1):
            candidate = vector.copy()
            candidate[col] += sign*epsilon
            predictions.append(lib.vascular_flow_forward(candidate[:4], np.exp(candidate[4:]),
                parameters, .3, derivative=False, **options)['canonical_prediction'])
        numeric.append(((predictions[0]-predictions[1])/(2*epsilon)).ravel())
    np.testing.assert_allclose(result['jacobian'], np.array(numeric).T, atol=8e-10, rtol=2e-6)
    # Linear interpolation lets the final flow node affect the final interval.
    identity = lib.vascular_flow_forward(np.zeros(steps), initial, parameters, .3)
    assert np.any(identity['jacobian'][-2:, steps-1] != 0)
    np.testing.assert_array_equal(identity['jacobian'][:-2, steps-1], 0.)


def test_noiseless_forward_fit_uses_nonoracle_starts_and_full_observation_operator():
    parameters, steps, dt = BalloonParameters(), 32, .25
    parameters = replace(parameters, fixed=replace(parameters.fixed, P0=42., Q0=14.))
    basis = cosine_basis(steps, 5)
    coefficients = np.array([.025, .09, -.045, .025, -.018])
    initial = np.array([1.07, 1.035, .975])
    truth = lib.vascular_flow_forward(coefficients, initial, parameters, dt,
        temporal_basis=basis, substeps=8)['canonical_prediction']
    # Temporal centering plus a fixed chromophore-mixing transform must both
    # apply before residual selection, rather than centering scored rows only.
    temporal = np.eye(steps)-np.ones((steps, steps))/steps
    operator = np.kron(temporal, np.array([[1.1, .12], [-.08, .9]]))
    target = (operator@truth.ravel()).reshape(steps, 2)
    result = lib.fit_vascular_flow_relaxation(target, parameters, dt,
        temporal_basis=basis, hb_operator=operator, sd=[1., .4], max_nfev=200,
        optimizer_tolerance=1e-14)
    assert result['legal_witness'], result['starts']
    assert result['optimizer_converged'], result['starts']
    assert result['weighted_data_sse'] < 1e-9
    assert result['by_chromophore']['HbO']['nrmse'] < 1e-5
    assert result['by_chromophore']['HbR']['nrmse'] < 1e-5
    assert len(result['starts']) == 3
    assert result['states'].shape == (steps, 3)
    np.testing.assert_allclose(result['prediction'],
        (operator@result['canonical_prediction'].ravel()).reshape(steps, 2), atol=1e-13)
    assert result['initial_state'][0] != result['initial_state'][1]
    assert not result['incompatibility_certified']
    assert not result['global_lower_bound_certified']


def test_budget_exhaustion_retains_legal_witness_and_never_claims_convergence():
    parameters, steps = BalloonParameters(), 9
    target = np.full((steps, 2), .01)
    result = lib.fit_vascular_flow_relaxation(target, parameters, .25, max_nfev=1,
        starts=[dict(coefficients=np.zeros(steps), initial_state=np.ones(3))])
    assert result['legal_witness']
    assert not result['optimizer_converged']
    assert result['optimizer_status'] == 'evaluation_budget'
    assert result['evaluations'] == 1
    assert result['weighted_data_sse'] == pytest.approx(2*steps*.01**2)


def test_integration_and_domain_failures_do_not_certify_incompatibility():
    parameters, steps = BalloonParameters(), 9
    initial = np.array([1.02, 1.01, .99])
    integration = lib.fit_vascular_flow_relaxation(np.zeros((steps, 2)), parameters, .5,
        starts=[dict(coefficients=.04*np.sin(np.arange(steps)), initial_state=initial)],
        max_nfev=1, substeps=1, validation_substeps=2, integration_tolerance=1e-15)
    assert integration['physical_checks']['positive_fvpq']
    assert not integration['integration']['passed']
    assert not integration['legal_witness']
    assert not integration['incompatibility_certified']
    domain = lib.fit_vascular_flow_relaxation(np.zeros((steps, 2)), parameters, .25,
        starts=[dict(coefficients=np.zeros(steps), initial_state=[1., .5, 2.])], max_nfev=1)
    assert domain['status'] == 'failed_domain'
    assert domain['starts'][0]['domain_rejections'] == 1
    assert not domain['legal_witness']
    assert not domain['incompatibility_certified']


@pytest.mark.parametrize('predictions,invalid_fine,expected_coefficient', [
    ([[.6, .6], [.4, .4], [.1, .1]], 2, .1),
    ([[.49, .49], [0., .6]], None, 0.),
    ([[0., .6], [.49, .49]], None, .1),
])
def test_retained_joint_witness_survives_later_better_sse_and_integration_failure(
        monkeypatch, predictions, invalid_fine, expected_coefficient):
    """Exercise the candidate-retention path, independently of optimizer luck."""
    from types import SimpleNamespace
    import scipy.optimize

    steps = 4
    physical = dict(finite=True, positive_fvpq=True,
        oxygen_extraction_in_unit_interval=True, absolute_hb_nonnegative=True,
        hbr_not_above_hbt=True)
    fine_nodes = []

    def forward(coefficients, initial, parameters, dt, *, temporal_basis,
                substeps, derivative=True):
        node = int(round(10*coefficients[0]))
        prediction = np.tile(predictions[node], (steps, 1)).astype(float)
        if substeps == 8:
            fine_nodes.append(node)
            if node == invalid_fine:
                prediction += .02  # Better SSE, but incompatible 4/8-step replay.
        log_flow = temporal_basis@coefficients
        return dict(states=np.ones((steps, 3)), flow=np.exp(log_flow), log_flow=log_flow,
            canonical_prediction=prediction, physical_checks=physical,
            jacobian=np.ones((2*steps, 4)) if derivative else None)

    def minimize(objective, x0, **kwargs):
        for node in range(1, len(predictions)):
            candidate = x0.copy()
            candidate[0] = node/10
            objective(candidate)
        return SimpleNamespace(success=False, status=9, nit=len(predictions)-1,
                               message='controlled nonconverged search')

    monkeypatch.setattr(lib, 'vascular_flow_forward', forward)
    monkeypatch.setattr(scipy.optimize, 'minimize', minimize)
    result = lib.fit_vascular_flow_relaxation(np.zeros((steps, 2)), BalloonParameters(), .25,
        temporal_basis=np.ones((steps, 1)), max_nfev=len(predictions),
        starts=[dict(coefficients=np.zeros(1), initial_state=np.ones(3))])
    assert result['legal_witness'] and result['joint_hb_witness']
    assert not result['optimizer_converged']
    assert not result['incompatibility_certified']
    assert result['coefficients'][0] == pytest.approx(expected_coefficient)
    assert result['by_chromophore']['HbO']['nrmse'] <= .5
    assert result['by_chromophore']['HbR']['nrmse'] <= .5
    assert fine_nodes == list(range(len(predictions)))
    assert result['retained_candidates'] == len(predictions)
    assert result['validation_evaluations'] == len(predictions)
    assert result['starts'][0]['legal_witness']
    assert result['starts'][0]['joint_hb_witness']
    if invalid_fine is None:
        assert result['minimum_legal_weighted_data_sse'] < result['weighted_data_sse']


def test_unscored_values_are_isolated_and_output_does_not_read_files(monkeypatch):
    import builtins

    parameters, steps = BalloonParameters(), 12
    coefficients = .03*np.sin(np.arange(steps)*.4)
    target = lib.vascular_flow_forward(coefficients, [1., 1., 1.], parameters, .25)['canonical_prediction']
    mask = np.ones((steps, 2), bool)
    mask[:4] = False
    options = dict(score_mask=mask, sd=[.02, .01], max_nfev=25,
        starts=[dict(coefficients=np.zeros(steps), initial_state=np.ones(3))])
    first = lib.fit_vascular_flow_relaxation(target, parameters, .25, **options)
    altered = target.copy()
    altered[~mask] = np.nan

    def forbidden_open(*args, **kwargs):
        raise AssertionError('the array-only inference API must not read files')

    # Warmed numba code is already loaded; the fit itself needs no file access.
    monkeypatch.setattr(builtins, 'open', forbidden_open)
    second = lib.fit_vascular_flow_relaxation(altered, parameters, .25, **options)
    np.testing.assert_array_equal(first['prediction'], second['prediction'])
    assert first['weighted_data_sse'] == second['weighted_data_sse']
    assert first['starts'] == second['starts']


def test_flow_bounds_are_pointwise_for_nonidentity_basis_and_invalid_stages_raise():
    parameters, steps = BalloonParameters(), 12
    basis = 8*cosine_basis(steps, 2)
    target = np.column_stack((np.linspace(0, 2, steps), np.zeros(steps)))
    result = lib.fit_vascular_flow_relaxation(target, parameters, .25,
        temporal_basis=basis, log_flow_bounds=(-.1, .1), max_nfev=60,
        starts=[dict(coefficients=np.zeros(2), initial_state=np.ones(3))])
    assert np.max(result['log_flow']) <= .1+1e-10
    assert np.min(result['log_flow']) >= -.1-1e-10
    np.testing.assert_allclose(result['log_flow'], basis@result['coefficients'])
    for backend in ('python', 'numba'):
        with pytest.raises(FloatingPointError, match='Hb balance'):
            lib.vascular_flow_forward(np.zeros(steps), [1., .5, 2.], parameters, .25,
                                      numerical_backend=backend)
        with pytest.raises(FloatingPointError):
            lib.vascular_flow_forward(np.zeros(steps), [1.01, 1., 1.], parameters, 100.,
                                      substeps=1, numerical_backend=backend)


@pytest.mark.parametrize('options,match', [
    (dict(sd=[1., 0.]), 'sd'),
    (dict(score_mask=np.zeros((8, 2), bool)), 'score_mask'),
    (dict(temporal_basis=np.ones((7, 2))), 'temporal_basis'),
    (dict(hb_operator=np.eye(8)), 'hb_operator'),
    (dict(log_flow_bounds=[1., -1.]), 'bounds'),
    (dict(validation_substeps=4), 'validation_substeps'),
    (dict(max_nfev=True), 'max_nfev'),
    (dict(joint_hb_tolerance=0.), 'tolerances'),
    (dict(starts=[]), 'starts'),
])
def test_fit_rejects_invalid_array_contracts(options, match):
    with pytest.raises(ValueError, match=match):
        lib.fit_vascular_flow_relaxation(np.zeros((8, 2)), BalloonParameters(), .25, **options)


def test_hb_and_initial_state_shapes_are_explicit():
    with pytest.raises(ValueError, match='hb must have shape'):
        lib.fit_vascular_flow_relaxation(np.zeros((8, 3)), BalloonParameters(), .25)
    with pytest.raises(ValueError, match='initial_physical'):
        lib.vascular_flow_forward(np.zeros(8), np.ones(5), BalloonParameters(), .25)
