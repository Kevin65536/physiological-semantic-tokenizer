"""Exact-input empirical bounds must preserve scoring masks and G0 hierarchy."""
import numpy as np
import pytest

from experiments.scripts.audit_hb_supervision_ambiguity import (
    canonical, hierarchy_weights, max_shared_spread, normalized_error, pointwise_constant,
)
from src.analysis.dual_component_tokenizer import base_training_aggregate


def test_exact_numeric_hash_canonicalizes_only_signed_zero():
    assert canonical(np.array([0., -0., 1.])).tobytes() == canonical(np.array([-0., 0., 1.])).tobytes()
    assert canonical(np.array([0., 1.])).tobytes() != canonical(np.array([0., 1. + 1e-12])).tobytes()
    with pytest.raises(ValueError):
        canonical(np.array([np.nan]))


def test_masked_pointwise_optimum_uses_per_example_valid_time_denominator():
    targets = np.zeros((2, 2, 2, 2))
    targets[0, :, 1] = 10
    targets[1, :, 0] = 2
    masks = np.array([[[True, True], [True, True]], [[True, True], [False, False]]])
    weights = np.array([.5, .5])
    mean, support = pointwise_constant(targets, masks, weights)
    np.testing.assert_allclose(mean[:, 0], 4 / 3)
    np.testing.assert_allclose(mean[:, 1], 10)
    assert support.all()
    best = (normalized_error(targets, masks, mean, np.ones((2,2))) * weights[:,None,None]).sum()
    worse = (normalized_error(targets, masks, mean + .1, np.ones((2,2))) * weights[:,None,None]).sum()
    assert best < worse


def test_g0_hierarchy_weights_match_the_existing_aggregate():
    rows = [dict(dataset='a', subject='s1', record='r1', parent='p1'),
            dict(dataset='a', subject='s1', record='r2', parent='p2'),
            dict(dataset='a', subject='s2', record='r1', parent='p3'),
            dict(dataset='b', subject='s1', record='r1', parent='p4')]
    examples = np.array([[0,0,0],[0,0,1],[0,30,0],[1,0,0],[2,0,0],[3,0,0]])
    values = np.arange(24).reshape(6,2,2)
    weights = hierarchy_weights(examples, rows)
    assert np.isclose(weights.sum(), 1)
    actual = base_training_aggregate({'nmse': values}, examples, rows)['metrics']['nmse']['macro']
    np.testing.assert_allclose((values * weights[:,None,None]).sum(0), actual)


def test_different_unscored_targets_create_no_false_conflict():
    targets = np.zeros((2,2,3,2))
    targets[1,:,2] = 999
    mask = np.ones((2,3,2), bool)
    mask[1,2] = False
    assert max_shared_spread(targets,mask) == 0
    targets[1,:,0] = 1
    assert max_shared_spread(targets,mask) == 1
