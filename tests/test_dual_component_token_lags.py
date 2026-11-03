"""Small synthetic sequence checks; no local measured artifact is read."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("dual_lags", Path(__file__).parents[1] /
    "experiments/scripts/analyze_dual_component_token_lags.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_positive_lag_means_hb_follows_eeg_and_edges_are_excluded():
    eeg = np.zeros((2, 1, 10, 2), dtype=int)
    eeg[0, 0, 2, 1] = 5
    eeg[1, 0, 9, 1] = 99  # partial-context edge must not enter
    hb = np.zeros((2, 1, 10), dtype=int)
    hb[0, 0, 4] = 1
    counts = module.lag_counts(module.sparse_lag(eeg, hb, 2), hb, np.arange(2), 2)
    assert counts[1, 1] == 5
    assert counts.sum() == 5
    backwards = module.lag_counts(module.sparse_lag(eeg, hb, -1), hb, np.arange(2), 2)
    assert backwards[1, 0] == 5


def test_trial_permutation_preserves_strata_and_rois_together():
    groups = [np.array([0, 2, 4]), np.array([1, 3, 5])]
    order = module.permute_trials(groups, np.random.default_rng(17))
    for group in groups:
        assert set(order[group]) == set(group)
    assert len(np.unique(order)) == 6


def test_exact_expected_counts_match_all_permutations():
    eeg = np.zeros((2, 1, 10, 2), dtype=int)
    eeg[0, 0, 3, 0] = 2
    eeg[1, 0, 3, 1] = 4
    hb = np.zeros((2, 1, 10), dtype=int)
    hb[1, 0, 4] = 1
    sparse = module.sparse_lag(eeg, hb, 1)
    identity = module.lag_counts(sparse, hb, np.array([0, 1]), 2)
    swapped = module.lag_counts(sparse, hb, np.array([1, 0]), 2)
    expected = module.expected_counts(eeg, hb, [np.arange(2)], 1, 2)
    np.testing.assert_array_equal(expected, (identity + swapped) / 2)


def test_event_locked_pattern_is_retained_by_null_not_evidence_of_trial_coupling():
    eeg = np.zeros((12, 1, 10, 2), dtype=int)
    hb = np.zeros((12, 1, 10), dtype=int)
    for t in range(10):
        eeg[:, 0, t, t % 2] = 3
        hb[:, 0, t] = (t - 1) % 2
    observed = module.lag_counts(module.sparse_lag(eeg, hb, 1), hb, np.arange(12), 2)
    expected = module.expected_counts(eeg, hb, [np.arange(12)], 1, 2)
    assert module.mi_bits(observed) > 0.9
    np.testing.assert_array_equal(observed, expected)


def test_synthetic_delayed_pair_exceeds_shuffled_trials():
    rng = np.random.default_rng(29)
    x = rng.integers(0, 4, (200, 1, 10))
    eeg = np.eye(4, dtype=int)[x]
    hb = rng.integers(0, 4, x.shape)
    hb[:, :, 2:] = x[:, :, :-2]
    sparse = module.sparse_lag(eeg, hb, 2)
    observed = module.mi_bits(module.lag_counts(sparse, hb, np.arange(200), 4))
    null = [module.mi_bits(module.lag_counts(sparse, hb, rng.permutation(200), 4)) for _ in range(20)]
    assert observed > 1.9
    assert max(null) < .1


def test_holm_is_monotone_and_controls_family():
    np.testing.assert_allclose(module.holm([.01, .04, .2]), [.03, .08, .2])


def test_unknown_time_support_fails():
    with pytest.raises(ValueError, match="ten aligned"):
        module.sparse_lag(np.zeros((2, 1, 9, 2)), np.zeros((2, 1, 9), int), 1)


def test_subject_analysis_retains_true_delay_after_trial_null(tmp_path):
    rng = np.random.default_rng(73)
    x = rng.integers(0, 2, (60, 1, 10))
    eeg = (3 * np.eye(2, dtype=int)[x])[..., None]
    hb = rng.integers(0, 2, (60, 1, 10, 1))
    hb[:, :, 2:, 0] = x[:, :, :-2]
    metadata = [dict(record="session", condition="condition", session="session") for _ in x]
    cfg = dict(lags_seconds=[-4, 0, 4], primary_levels=[0, 0], primary_permutations=39,
               secondary_permutations=19, permutation_seed=19, pair_min_eeg_occurrences=10,
               pair_min_expected_count=5, pair_min_trials=10)
    result = module.analyze_subject((("synthetic", "subject", "validation"), eeg, hb, metadata, cfg, 17, str(tmp_path)))
    assert result["status"] == "completed"
    assert max(result["rows"], key=lambda r: r["excess_mi_bits"])["lag_s"] == 4
    assert result["pairs"][0]["conditional_probability"] == 1
    assert result["pairs"][0]["p_max_pairs_positive_lags"] <= .05


def test_cross_split_subject_is_rejected_before_array_access(tmp_path):
    (tmp_path / "voltage_summary.json").write_text(json.dumps(dict(status="completed", experiment_id="TK-DUAL-VOLTAGE-v2")))
    (tmp_path / "voltage_selection.json").write_text(json.dumps(dict(eeg_candidate="eeg_voltage", hb_candidate="hb_blocks",
        selection_split="validation", test_used=False)))
    for folder, split in (("voltage_cache", "validation"), ("voltage_test_cache", "test")):
        target = tmp_path / folder
        target.mkdir()
        row = dict(dataset="eeg_fnirs_single_trial", subject="same_subject", split=split, id=folder)
        (target / "manifest.json").write_text(json.dumps(dict(schema="dual_component_voltage_cache_v1",
            protected_data="forbidden", rows=dict(eeg=[row], hb=[row]))))
    with pytest.raises(ValueError, match="cross-split"):
        module.validate_parent(tmp_path)


def test_maximum_scan_adjusts_individual_lag_p_values():
    null = np.array([[0., 2.], [1., 1.], [2., 0.], [1., 1.]])
    omnibus, adjusted = module.maximum_test(np.array([3., .5]), null)
    # A null draw in the other column exceeds the observed standardized peak.
    assert omnibus == .4
    assert adjusted[1] == 1.


def test_heatmap_marginal_equalizes_lag_exposure_and_uses_fixed_word_rank():
    counts = np.array([[[90., 10.], [0., 0.]], [[1., 9.], [0., 0.]]])
    marginal, js, rank, _ = module.conditional_heatmap_values(counts, counts, np.array([1., 0.]))
    np.testing.assert_allclose(marginal[0], [.5, .5])
    np.testing.assert_array_equal(js[0], [0., 0.])
    np.testing.assert_allclose(rank[0], [.9, .1])
    assert np.isnan(marginal[1]).all()
    assert np.isnan(rank[1]).all()


def test_heatmap_null_divergence_is_in_bits_and_symmetric():
    first = np.array([[[1., 0.]]])
    second = np.array([[[0., 1.]]])
    _, js, _, _ = module.conditional_heatmap_values(first, second, np.arange(2))
    _, reverse, _, _ = module.conditional_heatmap_values(second, first, np.arange(2))
    np.testing.assert_array_equal(js, [[1.]])
    np.testing.assert_array_equal(js, reverse)
