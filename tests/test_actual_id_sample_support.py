"""Sample-size diagnostics keep the held-subject population and native IDs fixed."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def owner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    return importlib.import_module('run_actual_id_coarsening')


def test_added_positions_stay_in_encoded_training_crops(monkeypatch):
    module = owner(monkeypatch)
    rows = [dict(dataset='fixture', subject=s, split='train', modality='eeg',
                 inputs={'score': [dict(start_s=0, positions=[0, 1, 2])]}) for s in ('a', 'b')]
    original = np.array([[0, 0, 0], [1, 0, 0]])
    extra = module.additional_training_examples(original, rows, [(0, 0), (1, 0)], 2, 17)
    assert len(extra) == 2
    assert set(extra[:, 0]) == {0, 1}
    assert (extra[:, 1] == 0).all()
    assert (extra[:, 2] > 0).all()
    rows[1]['split'] = 'validation'
    with pytest.raises(ValueError, match='non-training'):
        module.additional_training_examples(original, rows, [(0, 0), (1, 0)], 2, 17)


def test_held_extra_targets_cannot_change_fitting_means(monkeypatch):
    module = owner(monkeypatch)
    subject = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    targets = np.array([1., 3., 6., 6., 5., 7., 1000., 1000.])
    bank = dict(examples=np.zeros((8, 3), int), subject=subject, fold=subject,
                fixed_band=np.zeros(8, int), background=np.zeros((8, 15), int),
                valid=np.ones((8, 15), bool), S=np.ones((8, 15), int),
                S_attributes=np.broadcast_to(targets[:, None, None], (8, 15, 10)))
    rows, error, _, _, _, counts = module.sample_support_readout(bank, 4, 1, True, 1, 'S')
    np.testing.assert_array_equal(rows, [2, 3])
    np.testing.assert_array_equal(error, 4.)
    assert counts['fitting_subjects'] == [0]
    assert counts['held_subjects'] == [1]
