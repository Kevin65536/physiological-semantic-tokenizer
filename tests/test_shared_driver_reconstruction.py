"""Independent rollout, masking and rank checks for the linear screen."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.inference import t3a_balloon_robust_ssm as core
from src.inference.shared_driver_reconstruction import (
    build_shared_driver_design, fit_shared_driver, replay_nonlinear_driver,
)


def test_linear_design_matches_small_nonlinear_driven_rollout():
    parameters = core.BalloonParameters()
    steps, dt = 65, .2
    design = build_shared_driver_design(parameters, steps, dt)
    driver = .0005*np.sin(np.arange(steps)*dt*.7)
    initial = np.array([.00001, .00002, -.00001, .00001, -.00001])
    coefficients = np.r_[driver, initial]
    linear = (design.canonical_design@coefficients).reshape(steps, 3)
    z = np.r_[initial[0], np.log1p(initial[1:])]
    output = []
    for t in range(steps):
        state = np.r_[driver[t], z[0], np.exp(z[1:])]
        output.append(core.observation_map(state, parameters))
        if t < steps-1:
            result = solve_ivp(lambda _, v: core.balloon_rhs(np.r_[driver[t], v], parameters)[1:],
                (0., dt), z, rtol=1e-10, atol=1e-12)
            assert result.success
            z = result.y[:, -1]
    np.testing.assert_allclose(linear, output, atol=3e-7, rtol=0.)


def test_only_driver_and_initial_states_are_free_and_transition_is_exact():
    design = build_shared_driver_design(core.BalloonParameters(), 20, .2)
    assert design.observation_design.shape == (60, 25)
    coefficients = np.random.default_rng(4).normal(0, .01, 25)
    vascular = np.einsum('tij,j->ti', design.state_design, coefficients)
    np.testing.assert_allclose(vascular[0], coefficients[-5:])
    np.testing.assert_allclose(vascular[1:], vascular[:-1]@design.transition.T
        + coefficients[:19, None]*design.input, atol=1e-15)
    # The last driver cannot affect any observed vascular sample.
    np.testing.assert_array_equal(design.state_design[:, :, 19], 0.)


def test_hidden_values_never_enter_fit_with_processed_mean_operator():
    steps = 16
    operator = np.kron(np.eye(steps)-np.ones((steps, steps))/steps, np.eye(3))
    design = build_shared_driver_design(core.BalloonParameters(), steps, .5,
        processed_mean_operator=operator)
    y = np.random.default_rng(10).normal(0, .01, (steps, 3))
    visible = np.ones_like(y, dtype=bool)
    visible[5:10, 1:] = False
    first = fit_shared_driver(y, design, visible=visible, observation_scale=np.array([1., 2., 3.]))
    y[~visible] = np.nan
    second = fit_shared_driver(y, design, visible=visible, observation_scale=np.array([1., 2., 3.]))
    np.testing.assert_array_equal(first['prediction'], second['prediction'])
    np.testing.assert_allclose(first['prediction'].mean(axis=0), 0., atol=1e-14)
    assert 0 <= first['effective_df'] <= visible.sum()


def test_zero_penalty_is_true_unregularized_residual_lower_bound():
    design = build_shared_driver_design(core.BalloonParameters(), 25, .2)
    y = np.random.default_rng(12).normal(0, .01, (25, 3))
    unregularized = fit_shared_driver(y, design, penalty=0., initial_penalty=1e6)
    expected = np.linalg.lstsq(design.observation_design, y.ravel(), rcond=1e-10)[0]
    np.testing.assert_allclose(unregularized['coefficients'], expected, rtol=1e-8, atol=1e-8)
    regularized = fit_shared_driver(y, design, penalty=.1)
    assert unregularized['data_sse'] <= regularized['data_sse']+1e-12
    assert unregularized['effective_df'] == pytest.approx(unregularized['rank'], abs=1e-7)
    assert unregularized['initial_penalty'] == 0.


def test_small_signal_invalid_solution_is_retained_and_flagged():
    design = build_shared_driver_design(core.BalloonParameters(), 16, .2)
    coefficients = np.r_[np.ones(16)*.05, np.array([0., .7, .6, .5, .4])]
    y = (design.observation_design@coefficients).reshape(16, 3)
    result = fit_shared_driver(y, design, penalty=0.)
    assert result['data_sse'] < 1e-15
    assert not result['physical_validity']['small_signal_valid']
    assert not result['physical_validity']['valid']


def test_invalid_shapes_and_nonfinite_visible_observations():
    p = core.BalloonParameters()
    for steps, dt in [(2, .1), (3.5, .1), (5, 0), (5, np.nan)]:
        with pytest.raises(ValueError):
            build_shared_driver_design(p, steps, dt)
    with pytest.raises(ValueError):
        build_shared_driver_design(p, 5, .1, processed_mean_operator=np.eye(5))
    design = build_shared_driver_design(p, 5, .1)
    y = np.zeros((5, 3))
    for kwargs in [dict(penalty=-1), dict(initial_penalty=-1), dict(rcond=0),
            dict(observation_scale=[1., 0., 1.]), dict(visible=np.zeros_like(y, dtype=bool)),
            dict(visible=np.ones_like(y))]:
        with pytest.raises(ValueError):
            fit_shared_driver(y, design, **kwargs)
    with pytest.raises(ValueError):
        fit_shared_driver(y[:, :2], design)
    y[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_shared_driver(y, design)


def test_nonlinear_replay_matches_small_signal_design_and_retains_stage_checks():
    p = core.BalloonParameters()
    steps, dt = 40, .25
    r = .0005*np.sin(np.arange(steps)*dt*.5)
    initial_delta = np.array([0., .00002, -.00001, .00001, -.00001])
    initial = initial_delta+np.array([0., 1., 1., 1., 1.])
    design = build_shared_driver_design(p, steps, dt)
    expected = (design.canonical_design@np.r_[r, initial_delta]).reshape(steps, 3)
    replay = replay_nonlinear_driver(r, initial, p, dt)
    assert replay['status'] == 'completed'
    assert replay['failure'] is None
    assert replay['checks']['all_intermediate_valid']
    assert len(replay['intermediate_checks']) == 1+(steps-1)*4*5
    np.testing.assert_allclose(replay['canonical_prediction'], expected, atol=3e-7, rtol=0.)
    np.testing.assert_array_equal(replay['states'][:, 0], r)


def test_nonlinear_replay_preserves_linear_signal_flow_dynamics_at_large_amplitude():
    # s and f form an exactly linear subsystem even when v/p/q are nonlinear.
    p = core.BalloonParameters()
    steps, dt = 50, .2
    r = .04+.03*np.sin(np.arange(steps)*dt*.8)
    initial_delta = np.array([.005, .08, .05, .04, .02])
    design = build_shared_driver_design(p, steps, dt)
    linear = np.einsum('tij,j->ti', design.state_design, np.r_[r, initial_delta])
    replay = replay_nonlinear_driver(r, initial_delta+np.array([0., 1., 1., 1., 1.]), p, dt)
    assert replay['status'] == 'completed'
    np.testing.assert_allclose(replay['states'][:, 1], linear[:, 0], atol=2e-9, rtol=0.)
    np.testing.assert_allclose(replay['states'][:, 2], 1+linear[:, 1], atol=2e-9, rtol=0.)


@pytest.mark.parametrize('initial', [[0., 0., 1., 1., 1.], [0., 1., 1., 1., 4.]])
def test_nonlinear_replay_rejects_initial_domain_without_clipping(initial):
    replay = replay_nonlinear_driver(np.zeros(5), initial, core.BalloonParameters(), .25)
    assert replay['status'] == 'failed_domain'
    assert replay['failure']['stage'] == 'initial'
    assert replay['checks']['completed_samples'] == 0
    assert np.isnan(replay['states']).all()
    np.testing.assert_array_equal(replay['intermediate_checks'][0]['state'][1:], initial)


def test_nonlinear_replay_rejects_midway_invalid_flow_and_retains_prefix():
    replay = replay_nonlinear_driver(np.full(20, -10.), [0., 1., 1., 1., 1.],
        core.BalloonParameters(), .25)
    assert replay['status'] == 'failed_domain'
    assert replay['failure']['stage'] != 'initial'
    assert 0 < replay['checks']['completed_samples'] < 20
    assert not replay['checks']['all_intermediate_valid']
    completed = replay['checks']['completed_samples']
    assert np.isfinite(replay['states'][:completed]).all()
    assert np.isnan(replay['states'][completed:]).all()
    assert not replay['intermediate_checks'][-1]['valid']


@pytest.mark.parametrize('kwargs', [dict(driver=[]), dict(driver=[np.nan]),
    dict(initial_state=[1., 1.]), dict(dt=0.), dict(substeps=0), dict(substeps=1.5)])
def test_nonlinear_replay_input_contract(kwargs):
    args = dict(driver=np.zeros(3), initial_state=[0., 1., 1., 1., 1.],
                parameters=core.BalloonParameters(), dt=.25)
    args.update(kwargs)
    with pytest.raises(ValueError):
        replay_nonlinear_driver(**args)
