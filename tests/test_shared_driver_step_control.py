"""Same-objective step control: nonlinear residual oscillation and real ODE checks."""
import numpy as np
import pytest

from src.inference import shared_driver_reconstruction as fit
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def oscillator(monkeypatch, *, reject_extra=False, worse_extra=False):
    """Nonzero-residual least squares, with a coupled nuisance/global direction.

    At zero, the residual-Hessian term is .98 whereas reduced GN curvature is
    100/101. Consequently undamped full GN alternates with factor near -.99.
    """
    original = fit._nonlinear_trial_objective
    calls = []

    def objective(x, p, *args, **kwargs):
        derivative = args[8]
        if reject_extra and len(calls) == 2 and not derivative:
            calls.append(None)
            raise FloatingPointError('synthetic interpolation domain rejection')
        result = original(x, p, *args, **kwargs)
        z = np.log(p.fixed.neurovascular_gain)
        r = np.r_[z+x[0], np.sqrt(.98)*(1+.5*z*z), 10*x]
        j = np.zeros((len(r), len(x)))
        j[0, 0] = 1
        j[2:] = 10*np.eye(len(x))
        b = np.r_[1., np.sqrt(.98)*z, np.zeros(len(x))]
        if worse_extra and len(calls) == 2 and not derivative:
            r = 10*r
        result.update(residual=r, objective=float(r@r))
        if derivative:
            result.update(jacobian=j, parameter_jacobian=b/p.fixed.neurovascular_gain)
        calls.append(dict(x=x.copy(), z=z, r=r, j=j, b=b, derivative=derivative))
        return result

    monkeypatch.setattr(fit, '_nonlinear_trial_objective', objective)
    return calls


def solve(**kwargs):
    batch = kwargs.pop('batch', 1)
    defaults = dict(parameter_name='neurovascular_gain', parameter_bounds=(.5, 2.),
        starts=[dict(parameter_value=np.exp(.05), driver=np.zeros((batch, 3)),
                     initial_state=np.tile([0., 1., 1., 1., 1.], (batch, 1)))],
        max_evaluations=240, max_iterations=40, gradient_tolerance=1e-7,
        penalty=0., initial_penalty=0., record_trace=True)
    defaults.update(kwargs)
    return fit.fit_nonlinear_shared_parameter(np.zeros((batch, 3, 3)),
        BalloonParameters(), .1, **defaults)


def test_nonlinear_nonzero_residual_oscillation_is_resolved(monkeypatch):
    oscillator(monkeypatch)
    legacy = solve()
    adaptive = solve(step_control='quadratic_interpolation')
    assert not legacy['converged']
    assert adaptive['converged']
    assert adaptive['objective'] == pytest.approx(.98, abs=1e-12)
    assert adaptive['parameter_value'] == pytest.approx(1., abs=1e-6)
    assert adaptive['optimization_evaluations'] < legacy['optimization_evaluations']
    assert adaptive['starts'][0]['interpolation_counts']['selected'] > 0
    z = np.log([t['parameter_value'] for t in legacy['trace'][-10:]])
    assert np.all(z[1:]*z[:-1] < 0)
    assert np.all(np.diff([t['objective'] for t in adaptive['trace']]) <= 0)


def test_full_arrow_prediction_includes_cross_terms_and_one_group_prior(monkeypatch):
    calls = oscillator(monkeypatch)
    result = solve(batch=2, step_control='quadratic_interpolation',
                   parameter_prior_mean=1., parameter_prior_log_sd=10., max_iterations=1)
    initial, proposed = calls[0], calls[2]
    dx, dz = proposed['x']-initial['x'], proposed['z']-initial['z']
    r = np.r_[initial['r'], initial['r'], initial['z']/10]
    jd = np.r_[initial['j']@dx+initial['b']*dz,
               initial['j']@dx+initial['b']*dz, dz/10]
    expected = r@r-(r+jd)@(r+jd)
    diagnostic = result['trace'][-1]['step_control_diagnostics']
    assert diagnostic['predicted_reduction'] == pytest.approx(expected, abs=1e-14)
    assert diagnostic['gain_ratio'] < .25
    assert diagnostic['interpolation_status'] == 'selected'
    assert result['parameter_prior_cost'] == pytest.approx((np.log(result['parameter_value'])/10)**2)


def test_extra_candidate_reserves_final_derivative_budget(monkeypatch):
    oscillator(monkeypatch)
    result = solve(step_control='quadratic_interpolation', max_evaluations=3)
    assert result['optimization_evaluations'] == 3
    assert result['starts'][0]['interpolation_counts']['budget_skips'] == 1
    assert result['trace'][-1]['accepted_fraction'] == 1
    assert result['trace'][-1]['step_control_diagnostics']['interpolation_status'] == 'budget_skip'


def test_interpolation_domain_failure_keeps_original_feasible_candidate(monkeypatch):
    oscillator(monkeypatch, reject_extra=True)
    result = solve(step_control='quadratic_interpolation', max_evaluations=4)
    attempt = result['starts'][0]
    assert result['optimization_evaluations'] == 4
    assert attempt['domain_rejections'] == 1
    assert attempt['interpolation_counts']['domain_rejections'] == 1
    assert attempt['evaluation_failures'][0]['derivative'] is False
    assert result['trace'][-1]['accepted_fraction'] == 1
    assert result['objective'] < .05**2+.98*(1+.5*.05**2)**2


def real_problem():
    p = BalloonParameters()
    r = .025*np.sin(np.arange(10)*.35)
    initial = np.array([.01, 1.02, 1.01, 1.015, .99])
    y = fit.nonlinear_driver_forward(r, initial, p, .2)['canonical_prediction']
    return p, np.stack([y, .9*y])


@pytest.mark.parametrize('parameter_name',['neurovascular_gain','tau'])
@pytest.mark.parametrize('coordinates',['independent_logs','positive_hb'])
def test_real_ode_flow_and_parameter_prior_cost_and_stationarity(parameter_name,coordinates):
    p, y = real_problem()
    reference=p.fixed.neurovascular_gain if parameter_name=='neurovascular_gain' else p.free.tau
    result = fit.fit_nonlinear_shared_parameter(y, p, .2,
        parameter_name=parameter_name, parameter_bounds=(.1, 8.),
        parameter_prior_mean=reference, parameter_prior_log_sd=.8,
        flow_prior_weight=.02, flow_prior_log_sd=.7, penalty=.003,
        initial_penalty=.02, driver_amplitude_weight=.01,
        max_evaluations=240, max_iterations=50, gradient_tolerance=1e-6,
        step_control='quadratic_interpolation', initial_coordinates=coordinates,record_trace=True)
    assert result['converged']
    assert result['projected_scaled_gradient_inf_norm'] <= 1e-6
    assert result['objective'] == pytest.approx(result['weighted_data_sse']+
        .003*result['roughness']+result['initial_state_penalty_cost']+
        result['driver_amplitude_cost']+result['flow_prior_cost']+result['parameter_prior_cost'])
    assert result['flow_prior_cost'] > 0
    assert result['optimization_evaluations'] <= 240


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        np.testing.assert_equal(a, b)


def test_default_and_explicit_armijo_exact_all_fields():
    p, y = real_problem()
    kw = dict(parameter_name='tau', parameter_bounds=(.3, 8.),
              max_evaluations=40, max_iterations=8, record_trace=True, return_jacobian=True)
    assert_same(fit.fit_nonlinear_shared_parameter(y, p, .2, **kw),
                fit.fit_nonlinear_shared_parameter(y, p, .2, step_control='armijo', **kw))


def test_unknown_step_control_rejected():
    with pytest.raises(ValueError, match='step_control'):
        solve(step_control='automatic')


def test_higher_interpolation_objective_does_not_replace_original(monkeypatch):
    oscillator(monkeypatch, worse_extra=True)
    result = solve(step_control='quadratic_interpolation', max_evaluations=4)
    assert result['starts'][0]['interpolation_counts']['not_selected'] == 1
    assert result['trace'][-1]['accepted_fraction'] == 1.
    assert result['trace'][-1]['step_control_diagnostics']['interpolation_status'] == 'not_selected'


def test_bound_intersection_keeps_original_projected_stationarity(monkeypatch):
    oscillator(monkeypatch)
    result = solve(step_control='quadratic_interpolation', parameter_bounds=(1.005, 1.1))
    assert result['converged']
    assert result['parameter_value'] == pytest.approx(1.005, abs=1e-14)
    assert result['boundary_status'] == 'LOWER'
    assert result['projected_scaled_gradient_inf_norm'] <= 1e-7
    assert result['projected_log_parameter_gradient'] == 0.
    assert result['log_parameter_gradient'] > 0.
    assert all(1.005-1e-14 <= p['parameter_value'] <= 1.1 for p in result['trace'])
