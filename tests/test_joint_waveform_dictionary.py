"""Synthetic contracts for the standalone integer waveform dictionary."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts.run_joint_waveform_dictionary import preserve_record, validate_config
from src.tokenizers.supervised_waveform_dictionary import (
    SupervisedWaveformDictionary, load_model, nearest_masked_waveform_ids,
)


def model():
    rng = np.random.default_rng(2)
    return SupervisedWaveformDictionary(
        dict(schema='supervised_waveform_dictionary_v1', modality='hb', codebook_size=4),
        {branch: np.tile(np.eye(2)[None] * .5, (151, 1, 1)).astype(complex) for branch in ('S', 'O')},
        {branch: np.zeros(40) for branch in ('S', 'O')},
        {branch: np.eye(40) for branch in ('S', 'O')},
        {branch: rng.normal(size=(4, 40)) for branch in ('S', 'O')},
    )


def test_saved_integer_sequences_decode_without_observations(tmp_path):
    original = model()
    observations = np.random.default_rng(4).normal(size=(3, 15, 40))
    ids = original.encode(observations)
    expected = original.decode_ids(ids)
    checkpoint = tmp_path / 'model.npz'
    original.save(checkpoint)
    del original, observations
    restored = load_model(checkpoint)
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(restored.decode_ids(ids)[branch], expected[branch])
    assert restored.decode_ids(ids)['S'].shape == (3, 15, 40)


def test_decoder_rejects_invalid_id_coordinates():
    tokenizer = model()
    integer = np.zeros((2, 15), dtype=np.int32)
    for invalid in (integer.astype(float), integer[:, :14], integer - 1, integer + 4):
        with pytest.raises(ValueError):
            tokenizer.decode_ids({'S': invalid, 'O': integer})


def test_config_drift_rejected_before_signal_reads():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_dictionary_v1.yaml').read_text())
    validate_config(cfg)
    for field, invalid in [('context_s', 60), ('token_s', 1), ('channels', {'eeg': 1, 'hb': 1}),
                           ('codebook_sizes', [0]), ('output_pca_dim', {'eeg': 401, 'hb': 32})]:
        modified = copy.deepcopy(cfg)
        modified['model'][field] = invalid
        with pytest.raises(ValueError):
            validate_config(modified)


def test_resume_preserves_retained_config(tmp_path):
    path = tmp_path / 'identity.json'
    preserve_record(path, {'seed': 29})
    preserve_record(path, {'seed': 29})
    with pytest.raises(ValueError):
        preserve_record(path, {'seed': 31})
    assert '29' in path.read_text()


def test_full_waveform_oracle_uses_dimensions_outside_encoder_projection():
    targets = np.array([[0., 9.]])
    atoms = np.array([[0., 0.], [1., 9.]])
    ids, mse, count = nearest_masked_waveform_ids(targets, np.ones_like(targets, bool), atoms,
                                                 target_block_size=1, atom_block_size=1)
    # Projection onto the first coordinate alone would choose ID 0.
    np.testing.assert_array_equal(ids, [1])
    np.testing.assert_allclose(mse, [.5])
    np.testing.assert_array_equal(count, [2])


def test_masked_waveform_oracle_matches_brute_force_with_partial_and_empty_support():
    rng = np.random.default_rng(114)
    targets = rng.normal(size=(7, 6))
    atoms = rng.normal(size=(11, 6))
    atoms[-1] = atoms[2]
    mask = rng.uniform(size=targets.shape) > .4
    mask[0] = False
    mask[1] = True
    targets[~mask] = np.nan
    ids, mse, count = nearest_masked_waveform_ids(targets, mask, atoms,
                                                 target_block_size=3, atom_block_size=4)
    for i in range(len(targets)):
        if not mask[i].any():
            assert ids[i] == 0 and np.isnan(mse[i])
            continue
        costs = ((targets[i, mask[i]]-atoms[:, mask[i]])**2).mean(1)
        assert ids[i] == np.argmin(costs)
        np.testing.assert_allclose(mse[i], costs[ids[i]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(count, mask.sum(1))


def test_masked_waveform_oracle_avoids_large_offset_distance_cancellation():
    targets = np.array([[1e10, 1e10 + 1.]])
    atoms = np.array([[1e10 + 1., 1e10], [1e10, 1e10 + 1.]])
    ids, mse, _ = nearest_masked_waveform_ids(targets, np.ones_like(targets, bool), atoms)
    np.testing.assert_array_equal(ids, [1])
    np.testing.assert_array_equal(mse, [0.])


def test_changed_time_unit_requires_explicit_diagnostic_and_replays(tmp_path):
    tokenizer = model()
    metadata = dict(tokenizer.metadata, atom_dim=20, atom_patches=30)
    means = {branch: np.zeros(20) for branch in ('S', 'O')}
    bases = {branch: np.eye(20) for branch in ('S', 'O')}
    atoms = {branch: np.ones((4, 20)) for branch in ('S', 'O')}
    with pytest.raises(ValueError):
        SupervisedWaveformDictionary(metadata, tokenizer.transfer, means, bases, atoms)
    metadata['interface_eligible'] = False
    diagnostic = SupervisedWaveformDictionary(metadata, tokenizer.transfer, means, bases, atoms)
    values = np.zeros((2, 15, 40))
    ids = diagnostic.encode(values)
    assert ids['S'].shape == (2, 30)
    path = tmp_path / 'diagnostic.npz'
    diagnostic.save(path)
    np.testing.assert_array_equal(load_model(path).decode_ids(ids)['S'], diagnostic(values)['S'])
