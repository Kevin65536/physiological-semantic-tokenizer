"""Observation-only Hb auxiliary inputs and nested linear diagnostic on synthetic data."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.data.dual_component_tokenizer import base_tokenizer_joint_hb

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('multimodal_hb', ROOT / 'experiments/scripts/diagnose_tokenizer_multimodal_hb.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_hb_auxiliary_uses_observations_without_reading_any_teacher_arrays(tmp_path):
    values = np.ones((300, 3, 2)) * np.array([2., 3., 4.])[None, :, None]
    observed = np.ones_like(values, bool)
    observed[1, 1, 1] = False
    values[2, 1, 0] = np.nan
    np.save(tmp_path / 'x.npy', values)
    np.save(tmp_path / 'observation_valid.npy', observed)
    # Position B is not a scored teacher target; C has no original train scale.
    row = dict(split='train', modality='hb', dataset='fixture', folder=str(tmp_path),
               positions=['A', 'B', 'C'], inputs={'score': [dict(start_s=0, positions=[0])]})
    scales = dict(positions={'hb': {'fixture': {'A': {'scale': 2}, 'B': {'scale': 3}}}})
    x, mask = base_tokenizer_joint_hb([row], dict(row=0, start_s=0), scales)
    np.testing.assert_array_equal(x[:, 0], 1.)
    assert mask[:, 0].all() and mask[0, 1].all()
    assert not mask[1, 1, 1] and not mask[2, 1, 0]
    assert not mask[:, 2].any() and not x[:, 2].any()
    assert np.isfinite(x).all()


def test_hb_rejects_split_and_crop_before_any_array_read(monkeypatch):
    row = dict(split='validation', modality='hb', inputs={'score': [dict(start_s=0)]})
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unexpected array read'))
    with pytest.raises(ValueError, match='before arrays'):
        base_tokenizer_joint_hb([row], dict(row=0, start_s=0), {})
    row['split'] = 'train'
    with pytest.raises(ValueError, match='outside frozen'):
        base_tokenizer_joint_hb([row], dict(row=0, start_s=30), {})


def test_signed_lag_and_nested_ridge_synthetic_null_and_positive_control():
    result = runner.synthetic()
    assert result['status'] == 'passed'
    assert result['moment_MSE_matches_direct']
    cfg = {'lag_seconds': [-1, 0, 1]}
    columns = runner.arm_columns(cfg)
    assert set(columns['local_hb']) < set(columns['local_plus_other_hb'])
    assert set(columns['local_plus_other_hb']) < set(columns['local_plus_other_hb_eeg'])


def test_eeg_power_summary_preserves_band_and_observation_support():
    t = np.arange(6000) / 200
    values = np.c_[np.sin(2 * np.pi * 10 * t), np.ones(6000) * 1e6]
    band, mask = runner.eeg_power_summary(values, np.array([True, False]), [[1, 4], [8, 13], [30, 45]])
    np.testing.assert_allclose(band[:, 1], np.log1p(.5), atol=1e-12)
    assert np.max(abs(band[:, [0, 2]])) < 1e-20 and mask.all()
    values[:, 1] = -1e6
    second, _ = runner.eeg_power_summary(values, np.array([True, False]), [[1, 4], [8, 13], [30, 45]])
    np.testing.assert_array_equal(second, band)
