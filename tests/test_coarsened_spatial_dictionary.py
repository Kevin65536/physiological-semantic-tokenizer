"""A coarse ID must fully determine its frozen representative waveform."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.tokenizers.coarsened_spatial_dictionary import CoarsenedSpatialDictionary, supported_remap


def test_remap_uses_supported_train_anchors_and_maps_unseen_IDs():
    atoms = np.array([[0., 0.], [1., 0.], [0., 1.], [1.1, 0.], [0., 1.1]])
    counts = np.array([0, 30, 20, 2, 0])
    subjects = np.array([0, 3, 3, 1, 0])
    semantic = np.array([[1., 1., 1.], [1., 2., 1.], [2., 1., 1.], [1., 2., 1.], [0., 0., 0.]])
    mapping, anchors = supported_remap(atoms, counts, subjects, semantic, 3, .75)
    assert anchors.tolist() == [0, 1, 2]
    assert mapping.tolist() == [0, 1, 2, 1, 2]


def test_integer_only_decode_lifts_coarse_ID_without_original_ID_side_input():
    class Source:
        def decode_ids(self, ids, dataset):
            assert dataset == 'study'
            return {key: value[..., None].repeat(400, axis=-1) for key, value in ids.items()}
    tables = {(b, band): np.array([0, 1, 1, 2]) for b in 'SO' for band in range(5)}
    representatives = {(b, band): np.array([0, 1, 3]) for b in 'SO' for band in range(5)}
    encoder = SimpleNamespace(codebook_size=4)
    metadata = dict(codebook_size=3, positions={'study': ['A', 'B']})
    model = CoarsenedSpatialDictionary(Source(), encoder, metadata, tables, representatives)
    fine = {b: np.full((1, 2, 15), 2, np.int32) for b in 'SO'}
    coarse = model.remap(fine, 'study')
    np.testing.assert_array_equal(coarse['S'], 1)
    np.testing.assert_array_equal(model.decode_ids(coarse, 'study')['S'], 1)
    with pytest.raises(ValueError, match='in-vocabulary'):
        model.decode_ids(fine | {'S': np.full((1, 2, 15), 3)}, 'study')
