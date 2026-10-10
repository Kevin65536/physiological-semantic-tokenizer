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
