"""Small analytic checks for the educational renderer; no retained data reads."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest


@pytest.fixture(scope="module")
def renderer():
    path = Path(__file__).resolve().parents[1] / "experiments/scripts/render_shared_driver_reconstruction.py"
    spec = importlib.util.spec_from_file_location("fitting_explainer_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gradient_flow_retains_unexplainable_residual_and_null_coordinate(renderer):
    a = np.diag([2., 1., 0.])
    b = np.array([1., 2., 3.])
    result = renderer.fitting_gradient_flow(a, b, frames=21)
    t = result["clock"]
    # Independent scalar ODE solutions, including an unobservable coordinate.
    expected = np.column_stack((.5*(1-np.exp(-4*t)), 2*(1-np.exp(-t)), np.zeros_like(t)))
    np.testing.assert_allclose(result["coordinates"], expected, atol=1e-13)
    assert result["rank"] == 2
    assert np.all(np.diff(result["sse"]) <= 1e-12)
    assert result["sse"][-1] == pytest.approx(9.)
    np.testing.assert_allclose(result["prediction"][0], np.zeros(3))


def test_extra_path_matches_independent_joint_ridge_and_refits_base(renderer):
    a = np.array([[1.], [0.], [0.]])
    c = np.array([[1.], [1.], [0.]])
    b = np.array([2., 3., 4.])
    result = renderer.fitting_component_path(a, b, c, frames=7)
    for j, lam in enumerate(result["penalty"]):
        if np.isinf(lam):
            np.testing.assert_allclose(result["prediction"][j], [2., 0., 0.])
            continue
        joint = np.vstack((np.column_stack((a, c)), [0., np.sqrt(lam)]))
        solution = np.linalg.lstsq(joint, np.r_[b, 0.], rcond=None)[0]
        np.testing.assert_allclose(result["prediction"][j], np.column_stack((a, c)) @ solution,
                                   atol=1e-12)
    # The new component has horizontal influence, so retaining the old base
    # coefficient would be wrong even though the final residual can be smaller.
    assert result["coordinates"][0, 0] == pytest.approx(2.)
    assert result["coordinates"][-1, 0] == pytest.approx(-1.)
    assert result["sse"][-1] == pytest.approx(16.)


def test_export_preserves_existing_artifact_directory(renderer, tmp_path):
    out = tmp_path / "retained"
    out.mkdir()
    evidence = out / "index.html"
    evidence.write_text("retained evidence")
    with pytest.raises(ValueError, match="fresh versioned export"):
        renderer.render_fitting_explainer(tmp_path / "nonexistent_run", out)
    assert evidence.read_text() == "retained evidence"


def test_parameter_case_selection_preserves_failures_and_distinct_subjects(renderer):
    import pandas as pd
    rows = []
    for subject, start in [('01', .1), ('09', .3), ('18', .6)]:
        for i, trial in enumerate([3, 7, 11, 15, 19, 23]):
            rows.append(dict(group=f'no_motion__subject_{subject}_o3', trial=trial,
                method='conditional_trained_gain', mode='full', status='completed',
                physical_valid=True, fine_physical_valid=True,
                nrmse_HbO=start+i*.01, nrmse_HbR=start+i*.01))
    rows[-1].update(status='failed', nrmse_HbO=99., nrmse_HbR=99.)
    cases, selection = renderer.select_parameter_explorer_cases(pd.DataFrame(rows))
    assert selection['planned'] == 18 and selection['failed'] == 1
    assert len({c['group'] for c in cases}) == 3
    assert cases[0]['trial'] == 3 and cases[-1]['trial'] == 19
    assert cases[1]['group'] == 'no_motion__subject_09_o3'
    with pytest.raises(ValueError, match='18 unique'):
        renderer.select_parameter_explorer_cases(pd.DataFrame(rows+[rows[0]]))


def test_parameter_sweep_replays_actual_forward_without_changing_eeg(renderer):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    from src.inference.t3a_balloon_robust_ssm import BalloonParameters
    n = 12
    driver = .01*np.sin(np.arange(n)*.2)
    initial = np.array([0., 1., 1., 1., 1.])
    parameters = BalloonParameters()
    baseline = nonlinear_driver_forward(driver, initial, parameters, .25, derivative=False)['canonical_prediction']
    result = renderer.parameter_response_frames(driver, initial, parameters, np.eye(3*n),
        baseline, np.ones(3), factors=[.75, 1., 1.25], backend='python')
    for response in result.values():
        for frame in response['frames']:
            assert frame['status'] == 'completed'
            np.testing.assert_allclose(np.asarray(frame['prediction'])[:, 0], baseline[:, 0], atol=1e-12)
        np.testing.assert_allclose(response['frames'][1]['prediction'], baseline, atol=1e-12)
        assert response['frames'][1]['hb_nrmse'] < 1e-12
        assert response['frames'][0]['hb_nrmse'] > 1e-8
    np.testing.assert_array_equal(initial, [0., 1., 1., 1., 1.])
    invalid = renderer.parameter_response_frames(driver, np.array([0., -1., 1., 1., 1.]), parameters,
        np.eye(3*n), baseline, np.ones(3), factors=[1.], backend='python')
    assert all(r['frames'][0]['status'] == 'domain_failed' for r in invalid.values())
    assert all(r['frames'][0]['prediction'] is None for r in invalid.values())


def test_parameter_export_checks_contract_before_array_reads(renderer, tmp_path):
    import json
    run = tmp_path / 'wrong'
    run.mkdir()
    (run / 'manifest.json').write_text(json.dumps({'execution': 'completed'}))
    (run / 'resolved_config.yaml').write_text('schema: protected_campaign\n')
    with pytest.raises(ValueError, match='Expected completed nonprotected'):
        renderer.render_parameter_explorer(run, tmp_path / 'fresh')
    out = tmp_path / 'retained'
    out.mkdir()
    with pytest.raises(ValueError, match='fresh versioned export'):
        renderer.render_parameter_explorer(run, out)


def test_boundary_evidence_requires_real_converged_hits_and_matching_trace(renderer):
    from copy import deepcopy
    selection = dict(status='completed', converged=True, parameter_name='tau',
        expected_training_trials=18, expected_starts=3, successful_starts=3,
        selected_training_start=0, parameter_value=.5)
    starts = [dict(status='completed', converged=True, start_index=i, start_tau=start,
        parameter_name='tau', parameter_bounds=[.5, 8.], boundary_status='LOWER', parameter_value=.5,
        objective=10., convergence_reason='projected_scaled_gradient', log_parameter_gradient=2.,
        projected_log_parameter_gradient=0., trace=[dict(iteration=0,parameter_value=start,objective=20.),
                                                    dict(iteration=1,parameter_value=.5,objective=10.)])
        for i,start in enumerate([1.,2.,4.])]
    evidence = renderer.parameter_boundary_evidence(selection, starts, [.5, 8.])
    assert evidence['first_boundary_iteration'] == 1 and evidence['training_windows'] == 18
    interior = deepcopy(starts)
    interior[1].update(boundary_status='INTERIOR', parameter_value=.7)
    with pytest.raises(ValueError, match='lower-bound claim'):
        renderer.parameter_boundary_evidence(selection, interior, [.5, 8.])
    starts[0]['trace'][-1]['parameter_value'] = .6
    with pytest.raises(ValueError, match='optimizer trace'):
        renderer.parameter_boundary_evidence(selection, starts, [.5, 8.])


def test_boundary_sweep_excludes_outside_nodes_before_solver(renderer, monkeypatch):
    from src.inference import shared_driver_reconstruction as core
    from src.inference.t3a_balloon_robust_ssm import BalloonParameters
    calls = []
    original = core.nonlinear_driver_forward
    def checked(driver, initial, parameters, dt, **kwargs):
        assert parameters.free.tau >= 2.
        calls.append(parameters.free.tau)
        return original(driver, initial, parameters, dt, **kwargs)
    monkeypatch.setattr(core, 'nonlinear_driver_forward', checked)
    result = renderer.parameter_response_frames(np.zeros(4), np.array([0.,1.,1.,1.,1.]),
        BalloonParameters(), np.eye(12), np.zeros((4,3)), np.ones(3), factors=[.5,1.,1.5],
        parameter_bounds={'tau':[2.,8.]}, backend='python')
    assert len(calls) == 16
    assert result['tau']['frames'][0]['status'] == 'outside_optimization_bounds'
    assert result['tau']['frames'][0]['prediction'] is None
    assert all(f['status']=='completed' for f in result['tau']['frames'][1:])
