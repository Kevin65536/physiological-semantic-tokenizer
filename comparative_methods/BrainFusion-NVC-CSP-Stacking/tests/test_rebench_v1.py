from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "rebench_v1.py"
SPEC = importlib.util.spec_from_file_location("brainfusion_downstream_rebench_v1", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture(autouse=True)
def small_synthetic_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _signals():
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(24) % 2
    eeg = torch.randn(24, 4, 160, generator=generator, dtype=torch.float64)
    hb = torch.randn(24, 6, 40, generator=generator, dtype=torch.float64)
    eeg[labels == 1, 0] *= 2
    hb[labels == 1, 0] *= 2
    subjects = [f"subject-{index // 4}" for index in range(24)]
    identities = [f"outer-train-{index}" for index in range(24)]
    return eeg, hb, labels, subjects, identities


def _profile():
    return {
        "task": "motor_imagery", "sampling_rate_hz": 20,
        "fnirs_sampling_rate_hz": 5, "inner_folds": 3,
        "feature_batch_size": 5, "nvc_pair_count": 6,
        "components_per_class": 1,
        "linear_svm_c_values": (0.1,), "rbf_svm_c_values": (),
        "random_forest_estimators": 4,
    }


def test_chunked_sufficient_statistics_match_dense_feature_algorithm():
    eeg, hb, labels, _, identities = _signals()
    estimator = module.create_estimator(_profile())
    chunked = estimator._new_features().fit(eeg, hb[:, :3], hb[:, 3:], labels, sample_ids=identities)
    dense = module.BrainFusionFeaturePipeline(
        nvc_config=estimator.nvc_config, csp_config=estimator.csp_config, nvc_pair_count=6,
    ).fit(eeg, hb[:, :3], hb[:, 3:], labels, sample_ids=identities)
    torch.testing.assert_close(chunked.selector.indices_, dense.selector.indices_)
    torch.testing.assert_close(chunked.selector.scores_, dense.selector.scores_, rtol=1e-8, atol=1e-8)
    actual = chunked.transform(eeg, hb[:, :3], hb[:, 3:])
    expected = dense.transform(eeg, hb[:, :3], hb[:, 3:])
    for view in module.VIEW_ORDER:
        torch.testing.assert_close(actual[view], expected[view], rtol=1e-7, atol=1e-7)


def test_complete_training_refit_and_every_supervised_oof_state_is_subject_isolated(tmp_path):
    eeg, hb, labels, subjects, identities = _signals()
    estimator = module.create_estimator(_profile()).fit(eeg, hb, labels, subjects=subjects, sample_ids=identities)
    audit = estimator.audit_state()["pipeline"]
    stacking = audit["stacking_state"]
    assert stacking["fit_sample_count"] == len(labels)
    assert estimator.fit_sample_ids_ == tuple(identities)
    assert estimator.pipeline.features.fit_sample_identity_sha256_ == module._validate_fit_identities(identities, len(labels))
    seen = set()
    for fold in stacking["inner_folds"]:
        assert not set(fold["train_subjects"]) & set(fold["validation_subjects"])
        assert not seen & set(fold["validation_subjects"])
        seen.update(fold["validation_subjects"])
        assert fold["feature_fit_audit"]["fit_sample_identity_sha256"] == fold["train_sample_identity_sha256"]
    assert seen == set(subjects)
    prediction = estimator.predict(eeg[:4], hb[:4])
    scores = estimator.decision_function(eeg[:4], hb[:4])
    assert prediction.shape == (4,) and scores.shape == (4, 2)
    estimator.save(tmp_path / "checkpoint")
    restored = module.BrainFusionFoldPipeline.load(tmp_path / "checkpoint")
    np.testing.assert_array_equal(restored.predict(eeg[:4], hb[:4, :3], hb[:4, 3:]), prediction)
    bounded_restore = module.RebenchBrainFusionEstimator.load(tmp_path / "checkpoint", profile=_profile())
    np.testing.assert_array_equal(bounded_restore.predict(eeg[:4], hb[:4]), prediction)
    assert isinstance(bounded_restore.pipeline.features, module._BatchedFeatures)
    assert bounded_restore.audit_state() == estimator.audit_state()


@pytest.mark.parametrize("task,n_classes", [("nback", 3), ("visual", 4)])
def test_multiclass_native_csp_and_scores(task, n_classes):
    eeg, hb, _, subjects, identities = _signals()
    labels = torch.arange(len(eeg)) % n_classes
    estimator = module.create_estimator({**_profile(), "task": task}).fit(eeg, hb, labels, subjects=subjects, sample_ids=identities)
    assert estimator.decision_function(eeg[:4], hb[:4]).shape == (4, n_classes)
    assert set(estimator.predict(eeg[:4], hb[:4])) <= set(range(n_classes))


def test_fit_rejects_event_ids_in_place_of_subject_partition_count_and_duplicate_samples():
    eeg, hb, labels, subjects, identities = _signals()
    estimator = module.create_estimator(_profile())
    with pytest.raises(ValueError, match="too few subjects"):
        estimator.fit(eeg, hb, labels, subjects=["one"] * len(labels), sample_ids=identities)
    with pytest.raises(ValueError, match="unique sample identity"):
        estimator.fit(eeg, hb, labels, subjects=subjects, sample_ids=["duplicate"] * len(labels))


@pytest.mark.parametrize("task", ["dsr", "refed_regression"])
def test_unsupported_tasks_do_not_emit_a_numeric_score(task):
    with pytest.raises(NotImplementedError):
        module.create_estimator({**_profile(), "task": task})


def test_partial_windows_use_only_each_oof_training_mean_and_retain_all_samples(tmp_path):
    eeg, hb, labels, subjects, identities = _signals()
    for index in range(len(hb)):
        hb[index] += (index // 4) * 0.7
    eeg_mask, hb_mask = torch.ones_like(eeg, dtype=torch.bool), torch.ones_like(hb, dtype=torch.bool)
    hb_mask[::5, :, -9:] = False
    eeg_mask[::7, :, -4:] = False
    expected_eeg, expected_hb = eeg.clone(), hb.clone()
    eeg[~eeg_mask], hb[~hb_mask] = float("nan"), float("nan")
    estimator = module.create_estimator(_profile()).fit(
        eeg, hb, labels, subjects=subjects, sample_ids=identities,
        eeg_mask=eeg_mask, fnirs_mask=hb_mask,
    )
    audit = estimator.audit_state()
    assert audit["pipeline"]["stacking_state"]["fit_sample_count"] == len(labels)
    assert audit["imputation"]["fnirs"]["partial_window_count"] == 5
    assert audit["imputation"]["fit_sample_identity_sha256"] == estimator.pipeline.features.fit_sample_identity_sha256_
    torch.testing.assert_close(estimator.imputation_state_["fnirs"]["channel_mean"],
                               torch.where(hb_mask, expected_hb, 0).sum((0, 2)) / hb_mask.sum((0, 2)))
    for fold in audit["pipeline"]["stacking_state"]["inner_folds"]:
        train = torch.tensor([subject in fold["train_subjects"] for subject in subjects])
        local_mask = hb_mask[train]
        expected = torch.where(local_mask, expected_hb[train], 0).sum((0, 2)) / local_mask.sum((0, 2))
        actual = fold["imputation_fit_audit"]
        np.testing.assert_allclose(actual["fnirs"]["channel_mean"], expected.numpy(), rtol=1e-12, atol=1e-12)
        assert actual["fit_sample_identity_sha256"] == fold["train_sample_identity_sha256"]
        assert actual["fit_sample_count"] == int(train.sum())
    before = estimator.decision_function(eeg, hb, eeg_mask=eeg_mask, fnirs_mask=hb_mask)
    eeg[~eeg_mask], hb[~hb_mask] = 1e20, -1e20
    np.testing.assert_array_equal(estimator.decision_function(eeg, hb, eeg_mask=eeg_mask, fnirs_mask=hb_mask), before)
    frozen_means = estimator.imputation_state_["fnirs"]["channel_mean"].clone()
    estimator.predict(eeg[:4], hb[:4] + 25, eeg_mask=eeg_mask[:4], fnirs_mask=hb_mask[:4])
    torch.testing.assert_close(estimator.imputation_state_["fnirs"]["channel_mean"], frozen_means)
    estimator.save(tmp_path / "partial_checkpoint")
    restored = module.RebenchBrainFusionEstimator.load(tmp_path / "partial_checkpoint", profile=_profile())
    np.testing.assert_array_equal(restored.decision_function(eeg, hb, eeg_mask=eeg_mask, fnirs_mask=hb_mask), before)
    assert restored.audit_state() == audit


def test_imputation_does_not_invent_an_entirely_unobserved_training_channel():
    eeg, hb, labels, subjects, identities = _signals()
    mask = torch.ones_like(hb, dtype=torch.bool)
    mask[:, 0] = False
    with pytest.raises(ValueError, match="channel with no recorded support"):
        module.create_estimator(_profile()).fit(eeg, hb, labels, subjects=subjects, sample_ids=identities, fnirs_mask=mask)


def test_nonfinite_recorded_input_cannot_be_hidden_by_imputation():
    eeg, hb, labels, subjects, identities = _signals()
    hb[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="recorded input contains non-finite"):
        module.create_estimator(_profile()).fit(eeg, hb, labels, subjects=subjects, sample_ids=identities)


def test_constant_eeg_and_hb_traces_are_zero_nvc_features_not_fabricated_waveforms():
    eeg, hb, _, _, _ = _signals()
    features = module.create_estimator(_profile())._new_features()
    # Nondegenerate pairs retain the exact source-kernel calculation.
    expected = module.BrainFusionFeaturePipeline(nvc_config=features.nvc_config)._nvc(eeg, hb[:, :3], hb[:, 3:])
    torch.testing.assert_close(features._nvc(eeg, hb[:, :3], hb[:, 3:]), expected, rtol=1e-12, atol=1e-12)
    eeg[0, 0] = 13.0
    hb[1, 1] = -4.0
    hb[2] = 0
    actual = features._nvc(eeg, hb[:, :3], hb[:, 3:]).reshape(24, 4, 6, 40)
    assert bool(torch.isfinite(actual).all())
    assert torch.count_nonzero(actual[0, 0]) == 0
    assert torch.count_nonzero(actual[1, :, 1]) == 0
    assert torch.count_nonzero(actual[2]) == 0
    assert features._last_nvc_degeneracy["undefined_pair_window_count"] == 6 + 4 + 24
    assert features._last_nvc_degeneracy["windows_with_undefined_pairs"] == 3


def test_whole_hb_channels_and_views_missing_in_train_or_prediction_are_retained(tmp_path):
    eeg, hb, labels, subjects, identities = _signals()
    eeg_mask, hb_mask = torch.ones_like(eeg, dtype=torch.bool), torch.ones_like(hb, dtype=torch.bool)
    # Every trial still has an identity and every channel has observed training
    # support elsewhere. Some individual trials have no Hb-view energy at all.
    hb_mask[0, 0] = False
    hb_mask[5] = False
    hb_mask[10, 2:4] = False
    eeg_mask[7, 1] = False
    eeg[~eeg_mask], hb[~hb_mask] = float("nan"), float("nan")
    estimator = module.create_estimator(_profile()).fit(
        eeg, hb, labels, subjects=subjects, sample_ids=identities,
        eeg_mask=eeg_mask, fnirs_mask=hb_mask,
    )
    audit = estimator.audit_state()
    assert audit["pipeline"]["stacking_state"]["fit_sample_count"] == 24
    assert audit["imputation"]["fnirs"]["whole_channel_window_count"] == 9
    assert audit["nvc_training_degeneracy"]["undefined_pair_window_count"] > 0
    for view in module.VIEW_ORDER:
        assert estimator.pipeline.features.csps[view].filters_.shape[0] == 2
    scores = estimator.decision_function(eeg, hb, eeg_mask=eeg_mask, fnirs_mask=hb_mask)
    assert bool(np.isfinite(scores).all())
    assert estimator.last_prediction_degeneracy_["undefined_pair_window_count"] > 0
    estimator.save(tmp_path / "whole_channel_missing")
    restored = module.RebenchBrainFusionEstimator.load(tmp_path / "whole_channel_missing", profile=_profile())
    np.testing.assert_array_equal(restored.decision_function(eeg, hb, eeg_mask=eeg_mask, fnirs_mask=hb_mask), scores)
    assert restored.audit_state() == audit
    # Entire evaluation Hb views can be absent without consulting evaluation
    # values or labels to manufacture a variable trace.
    missing_hb = torch.full_like(hb[:2], float("nan"))
    prediction = restored.predict(eeg[:2], missing_hb, eeg_mask=eeg_mask[:2], fnirs_mask=torch.zeros_like(missing_hb, dtype=torch.bool))
    assert prediction.shape == (2,)
    assert restored.last_prediction_degeneracy_["undefined_pair_window_count"] == 2 * 4 * 6


def test_zero_energy_csp_covariance_retains_zero_in_class_denominator():
    values = torch.randn(3, 4, 20, dtype=torch.float64)
    values[1] = 7
    covariance = module._BatchedFeatures._trial_covariances(values)
    assert torch.count_nonzero(covariance[1]) == 0
    expected = module.TorchCSP._trial_covariances(values[[0, 2]])
    torch.testing.assert_close(covariance[[0, 2]], expected)
