"""The sparse diagnostic ranks real isolated physical corrections and pays an exact quota."""

import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location(
    "event_slots",
    Path(__file__).resolve().parents[1] / "experiments/scripts/run_band_exchange_event_slots.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_physical_propagation_score_and_exact_selection():
    rng = np.random.default_rng(29)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    roles = np.array([[0, 1], [1, 0], [0, 1], [-1, -1]], dtype=np.int8)
    actual = rng.normal(size=(4, 2, 400))
    atomic = rng.normal(size=actual.shape)
    weights = np.array([0.5, 2.0, 3.0, 0.0])
    score = module.propagation_scores(actual, atomic, roles, np.repeat(cov[None], 2, axis=0), weights)
    anchors = np.array([0, 2])
    columns = np.linalg.solve(cov[np.ix_(anchors, anchors)], cov[anchors]).T
    delta = actual[0, 0] - atomic[0, 0]
    expected = np.square(columns[:, 0, None] * delta).sum(1) @ weights
    np.testing.assert_allclose(score[0, 0], expected)
    selected = module.select_slots(score, roles, 0.25)
    assert selected.sum() == 1
    assert not selected[3].any()
    assert module.select_slots(score, roles, 0).sum() == 0
    np.testing.assert_array_equal(module.select_slots(score, roles, 1), roles >= 0)


def test_equal_errors_break_ties_by_native_then_time():
    roles = np.zeros((2, 15), dtype=np.int8)
    selected = module.select_slots(np.ones_like(roles, float), roles, 0.1)
    expected = np.zeros_like(selected)
    expected[0, :3] = True
    np.testing.assert_array_equal(selected, expected)
