"""Output derivatives, same-observation controls and raw hidden-information tests."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.ndimage import gaussian_filter1d

from src.inference.observation_baselines import disjoint_native_bin_operator
from src.inference.ssm_output_audit import (
    OutputProblem, reference_parameters, visibility, fit_problem, hb_coordinates,
    worst_output_direction, output_profile,
)
from src.inference.shared_driver_rk4 import compiled_forward

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def setup():
    cfg = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_output_support_v1.yaml').read_text())
    base = yaml.safe_load((ROOT/cfg['source_config']).read_text())
    op = disjoint_native_bin_operator(np.arange(900)/10.)
    return cfg, reference_parameters(base), dict(eeg=op['eeg'], hb=op['model'])


def test_disjoint_native_support_prevents_hidden_information_return():
    t = np.arange(900)/10.
    op = disjoint_native_bin_operator(t)
    hidden = (t >= 30)&(t < 60)
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(900, 2)); other = raw.copy()
    other[hidden] += rng.normal(size=(hidden.sum(), 2))*1e5
    visible = np.r_[np.arange(120), np.arange(240, 360)]
    np.testing.assert_array_equal((op['raw']@raw)[visible], (op['raw']@other)[visible])
    model = np.column_stack((np.sin(np.arange(360)/40), np.cos(np.arange(360)/30)))
    native = np.column_stack([np.interp(t, np.arange(360)/4, model[:, j]) for j in range(2)])
    np.testing.assert_allclose(op['raw']@native, op['model']@model, atol=1e-14)
    for bad in (t[3:], t[:-4], t[::-1]):
        with pytest.raises(ValueError):
            disjoint_native_bin_operator(bad)


def test_native_clock_roundoff_uses_one_canonical_bin_partition():
    # Subtracting two large source-clock values can put nominal 60s just below
    # 60. The declared operator owns bin membership, including roundoff.
    t = (731.2+np.arange(900)/10.)-731.2
    op = disjoint_native_bin_operator(t)
    hidden = (op['native_groups'] >= 120)&(op['native_groups'] < 240)
    values = np.random.default_rng(3).normal(size=(len(t), 2))
    altered = values.copy(); altered[hidden] += 1000
    visible = np.r_[np.arange(120), np.arange(240, 360)]
    np.testing.assert_array_equal((op['raw']@values)[visible], (op['raw']@altered)[visible])


def test_runner_rejects_contract_boundary_and_duplicate_task_id(setup, tmp_path):
    from experiments.scripts import evaluate_ssm_output_support as runner
    cfg = deepcopy(setup[0]); cfg['data']['protected_data'] = 'allowed'
    path = tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='boundary'):
        runner.load_config(path)
    with pytest.raises(ValueError, match='canonical identity'):
        runner.run_tasks(setup[0], tmp_path, 'prepare', [dict(id='same', key='native1'), dict(id='same', key='native2')], 1)


@pytest.mark.parametrize('arm', ['free', 'chain_matched', 'chain_first'])
def test_exact_state_output_and_residual_derivatives(setup, arm):
    cfg, parameters, ops = setup
    p = OutputProblem(np.zeros((360, 3)), np.array([.025, .02, .01]), ops, parameters, cfg, arm=arm)
    x = p.initial(); x[:360] = .005*np.sin(np.arange(360)/30)
    x[360:p.cstart] = np.tile([.002, .01, -.01, .015, .02], len(p.segments))
    value = p.evaluate(x)
    for col in (4, 119, 120, 238, 359, 360, 362, 363, p.cstart+2):
        step = np.zeros_like(x); step[col] = 1e-6
        plus, minus = p.evaluate(x+step, False), p.evaluate(x-step, False)
        numeric = (plus['residual']-minus['residual'])/2e-6
        np.testing.assert_allclose(value['jacobian'][:, col], numeric, rtol=2e-4, atol=2e-6)
        for name in ('SSM_Hb', 'T', 'X', 'volume', 'p_minus_v', 'flow', 'driver'):
            _, j = p.output(value, name, dict(T=.02, X=.01))
            a, _ = p.output(plus, name, dict(T=.02, X=.01))
            b, _ = p.output(minus, name, dict(T=.02, X=.01))
            np.testing.assert_allclose(j[:, col], (a-b)/2e-6, rtol=2e-4, atol=3e-6)


def test_matched_boundary_priors_evaluate_same_physical_states(setup):
    cfg, parameters, ops = setup
    chain = OutputProblem(np.zeros((360, 3)), np.ones(3), ops, parameters, cfg)
    free = OutputProblem(chain.y, chain.sd, ops, parameters, cfg, arm='free')
    x = chain.initial(); x[:360] = .01*np.sin(np.arange(360)/45)
    x[360:365] = [.01, .02, .01, .025, .015]
    cv = chain.evaluate(x)
    fx = free.initial(); fx[:360] = x[:360]
    for k, (left, _) in enumerate(free.segments):
        physical = cv['states'][left, 1:]
        fx[360+5*k:365+5*k] = np.r_[physical[0], np.log(physical[1:])]
    fv = free.evaluate(fx)
    for key in ('states', 'physical', 'prediction', 'residual'):
        np.testing.assert_allclose(cv[key], fv[key], atol=1e-12, rtol=1e-11)
    assert cv['costs']['initial'] == pytest.approx(fv['costs']['initial'])
    np.testing.assert_array_equal(chain.basis, free.basis)
    np.testing.assert_array_equal(chain.regularizer[:, :360], free.regularizer[:, :360])


def test_hidden_entries_cannot_affect_fit_and_remaining_is_missing(setup):
    cfg, parameters, ops = setup
    rng = np.random.default_rng(22)
    y = rng.normal(size=(360, 3))*.0001
    mask = visibility('middle_Hb')
    cfg = deepcopy(cfg); cfg['solver']['max_iterations'] = 3
    p = OutputProblem(y, np.array([.025, .02, .01]), ops, parameters, cfg, visible=mask)
    other = y.copy(); other[~mask] = np.nan
    q = OutputProblem(other, p.sd, ops, parameters, cfg, visible=mask)
    a, b = fit_problem(p), fit_problem(q)
    np.testing.assert_array_equal(a['x'], b['x'])
    np.testing.assert_array_equal(a['value']['prediction'], b['value']['prediction'])
    assert np.isnan(a['remaining'][~mask]).all()
    np.testing.assert_allclose(a['value']['physical'][mask]+a['remaining'][mask], y[mask], atol=1e-18)


def test_model_coordinate_identity_and_initial_mismatch_decay(setup):
    _, parameters, _ = setup
    r = gaussian_filter1d(np.random.default_rng(7).normal(size=360), 8)*.01
    f = compiled_forward(r, np.array([.005, 1.01, .99, 1.02, 1.01]), parameters, .25, 8, True, False, state=True)
    s = f['states']; hb = f['canonical_prediction'][:, 1:]
    tx = hb_coordinates(hb)
    np.testing.assert_allclose(tx[:, 0], parameters.fixed.P0*(s[:, 4]-1), atol=1e-15)
    np.testing.assert_allclose(tx[:, 1], parameters.fixed.Q0*(s[:, 4]-s[:, 5]), atol=1e-15)
    d = s[:, 4]-s[:, 3]
    assert np.all(np.diff(d[:120]) < 1e-14)
    assert abs(d[120]) < abs(d[0])*1e-5
    assert abs(d[-1]) < 1e-12
    for col in (2, 362, 363):
        eps = 1e-6; rd = r.copy(); ri = np.array([.005, 1.01, .99, 1.02, 1.01])
        if col < 360:
            rd[col] += eps
        else:
            ri[col-360] *= np.exp(eps)
        other = compiled_forward(rd, ri, parameters, .25, 8, False, False)
        np.testing.assert_allclose(f['state_jacobian'][:, :, col], (other['states']-s)/eps, atol=2e-5, rtol=1e-3)


def test_profile_reoptimizes_component_and_honors_output_constraint(setup):
    cfg, parameters, ops = setup
    cfg = deepcopy(cfg); cfg['solver']['max_iterations'] = 80
    rng = np.random.default_rng(6)
    y = rng.normal(size=(360, 3))*np.array([.0001, .00005, .00002])
    problem = OutputProblem(y, np.array([.025, .02, .01]), ops, parameters, cfg, arm='free')
    ref = fit_problem(problem)
    assert ref['converged']
    diagnostic, rows = output_profile(problem, ref, 'T', dict(T=.02, X=.01), [-.1, .1])
    assert diagnostic['unknowns'] == 387
    assert len(rows) == 2 and all(r['converged'] for r in rows)
    assert all(abs(r['constraint_error']) <= cfg['solver']['constraint_tolerance'] for r in rows)
    assert all(r['output_rms_change'] >= abs(r['offset'])-1e-5 for r in rows)
    assert any(np.linalg.norm(r['x'][problem.cstart:]-ref['x'][problem.cstart:]) > 1e-6 for r in rows)
