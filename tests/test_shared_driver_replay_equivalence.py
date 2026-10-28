"""Replay keeps all stage checks while reusing frozen-parameter invariants."""
from dataclasses import replace

import numpy as np
import pytest

from src.inference import shared_driver_reconstruction as replay_module
from src.inference import t3a_balloon_robust_ssm as core


def _assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_same(a, b)
    else:
        np.testing.assert_equal(actual, expected)


def _vector_reference(checker, values, parameters):
    """Exercise the retained array arithmetic, including its original rest check."""
    values = np.asarray(values, float)
    singleton = values.shape == (1, 6)
    checks = checker(np.repeat(values, 2, axis=0) if singleton else values, parameters)
    if singleton:
        for name in ('flow_below_0_01_count', 'extraction_float_saturation_count'):
            checks[name] //= 2
    return checks


@pytest.mark.parametrize('flow', [1., .009, .01, 1e-200, 1e-308, 1e200, 1e308,
                                 0., -1., np.nan, np.inf])
@pytest.mark.parametrize('volume,deoxy', [(1., 1.), (0., 1.), (-1., 1.),
                                        (1., 0.), (1., -1.), (1., 4.),
                                        (1e308, 1e308), (np.nan, 1.), (1., np.inf)])
def test_single_stage_fields_match_original_array_checks(flow, volume, deoxy):
    parameters = core.BalloonParameters()
    values = np.array([[0., .02, flow, volume, volume, deoxy]])
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            expected = _vector_reference(core._physical_checks, values, parameters)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            with pytest.raises(type(exc), match=str(exc)):
                core._physical_checks(values, parameters)
        else:
            _assert_same(core._physical_checks(values, parameters), expected)


@pytest.mark.parametrize('e0', [1e-300, .32, np.nextafter(1., 0.)])
@pytest.mark.parametrize('p0,q0', [(1., .35), (1e308, 5e307)])
@pytest.mark.parametrize('flow', [1e-308, 1e-200, .001, 1., 1e200, 1e308])
def test_single_stage_extreme_parameters_preserve_values_and_errors(e0, p0, q0, flow):
    parameters = core.BalloonParameters()
    parameters = replace(parameters, fixed=replace(parameters.fixed, E0=e0, P0=p0, Q0=q0))
    parameters.validate()
    values = np.array([[0., 0., flow, 2., .8, 1.6]])
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            expected = _vector_reference(core._physical_checks, values, parameters)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            with pytest.raises(type(exc), match=str(exc)):
                core._physical_checks(values, parameters)
        else:
            _assert_same(core._physical_checks(values, parameters), expected)


@pytest.mark.parametrize('hemodynamics,viscoelastic_s', [('tak', 2.), ('inlet_balance', 0.)])
@pytest.mark.parametrize('substeps', [4, 8])
@pytest.mark.parametrize('scenario', ['valid', 'invalid_flow', 'invalid_hb', 'invalid_stage'])
def test_replay_preserves_every_stage_and_failure(monkeypatch, hemodynamics,
                                                 viscoelastic_s, substeps, scenario):
    parameters = core.BalloonParameters()
    driver = .025*np.sin(np.arange(8)*.23)
    initial = np.array([.02, 1.03, .98, .98, 1.01])
    if scenario == 'invalid_flow':
        initial[1] = 0.
    elif scenario == 'invalid_hb':
        initial[-1] = 4.
    elif scenario == 'invalid_stage':
        driver[:] = -100.
    options = dict(substeps=substeps, hemodynamics=hemodynamics,
                   viscoelastic_s=viscoelastic_s)
    actual = replay_module.replay_nonlinear_driver(driver, initial, parameters, .25, **options)
    original = core._physical_checks

    def original_checks(values, parameters, **unused):
        parameters.validate()
        return _vector_reference(original, values, parameters)

    monkeypatch.setattr(core, '_physical_checks', original_checks)
    expected = replay_module.replay_nonlinear_driver(driver, initial, parameters, .25, **options)
    _assert_same(actual, expected)
    if scenario == 'valid':
        assert actual['status'] == 'completed'
        assert len(actual['intermediate_checks']) == 1+(len(driver)-1)*substeps*5
    else:
        assert actual['status'] == 'failed_domain'
        assert not actual['intermediate_checks'][-1]['valid']
        if scenario == 'invalid_stage':
            assert actual['failure']['stage'] != 'initial'
        if hemodynamics == 'tak':
            states = np.array([record['state'] for record in actual['intermediate_checks']])
            np.testing.assert_array_equal(states[:, 3], states[:, 4])


@pytest.mark.parametrize('hemodynamics,viscoelastic_s', [('tak', 2.), ('inlet_balance', 0.)])
@pytest.mark.parametrize('name,value', [('E0', 0.), ('E0', 1.), ('E0', np.nan),
                                     ('tau', 0.), ('tau', -1.), ('tau', np.nan)])
def test_replay_validates_parameters_before_reusing_checks(hemodynamics, viscoelastic_s,
                                                          name, value):
    parameters = core.BalloonParameters()
    branch = 'fixed' if name == 'E0' else 'free'
    parameters = replace(parameters, **{branch: replace(getattr(parameters, branch), **{name: value})})
    with pytest.raises(ValueError):
        replay_module.replay_nonlinear_driver(np.zeros(3), [0., 1., 1., 1., 1.],
            parameters, .1, hemodynamics=hemodynamics, viscoelastic_s=viscoelastic_s)


def test_replay_computes_rest_once_and_stays_independent_of_compiled_forward(monkeypatch):
    from src.inference import shared_driver_rk4

    parameters = core.BalloonParameters()
    calls = []
    original = core.balloon_rhs

    def rest_rhs(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('independent replay must not call the compiled forward')

    monkeypatch.setattr(core, 'balloon_rhs', rest_rhs)
    monkeypatch.setattr(shared_driver_rk4, 'tak_forward', forbidden)
    monkeypatch.setattr(shared_driver_rk4, 'compiled_forward', forbidden)
    result = replay_module.replay_nonlinear_driver(np.zeros(4), [0., 1., 1., 1.], parameters, .1)
    assert result['status'] == 'completed'
    # One first-stage invariant, plus the complete trajectory's independent report.
    assert len(calls) == 2
