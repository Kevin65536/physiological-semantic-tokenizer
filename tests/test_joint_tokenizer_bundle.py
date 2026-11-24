import hashlib
import json

import numpy as np
import pytest

from experiments.scripts.apply_joint_tokenizer import apply
from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_encoder_input_scaling_matches_g0_per_position(dtype):
    from experiments.scripts.apply_joint_tokenizer import normalize_observations

    observation = np.linspace(-3, 4, 120, dtype=dtype).reshape(2, 30, 2)
    scales = [1.23456789123, 0.7451234567]
    expected = np.stack([row / scale for row, scale in zip(observation, scales)]).astype(np.float32)
    np.testing.assert_array_equal(normalize_observations(observation, scales), expected)


def test_four_branch_native_decode_uses_only_integer_ids_and_fixed_metadata(tmp_path):
    def entry(path):
        return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    scales = tmp_path / 'scales.json'
    scales.write_text(json.dumps({'positions': {m: {'study': {'site': {'scale': factor}}}
                                                for m, factor in [('eeg', 2.), ('hb', 3.)]}}))
    bundle = {'schema': 'joint_tokenizer_inference_v1', 'scales': entry(scales)}
    arrays = {}
    for m, width, colors in [('eeg', 400, 1), ('hb', 40, 2)]:
        z = np.arange(width, dtype=float)
        atoms = {b: np.array([z, z + step]) for b, step in [('S', 1.), ('O', 2.)]}
        metadata = dict(schema='supervised_waveform_dictionary_v1', modality=m, codebook_size=2)
        if m == 'hb':
            metadata.update(codebook_size=3, branch_codebook_sizes={'S': 2, 'O': 3})
            atoms['S'] = np.vstack([atoms['S'], z])
            atoms['O'] = np.vstack([atoms['O'], z + 4.])
        model = SupervisedWaveformDictionary(
            metadata,
            {b: np.zeros((15 * width // colors // 2 + 1, colors, colors)) for b in atoms},
            {b: z * 0 for b in atoms}, {b: np.eye(1, width) for b in atoms}, atoms)
        path = tmp_path / (m + '.npz')
        model.save(path)
        bundle[m] = dict(kind='dictionary', **entry(path))
        arrays[m + '_dataset'] = np.array(['study'])
        arrays[m + '_position'] = np.array(['site'])
        for b in atoms:
            arrays[m + '_' + b] = np.ones((1, 15), dtype=np.uint8)
    manifest = tmp_path / 'bundle.json'
    manifest.write_text(json.dumps(bundle))
    decoded = apply(manifest, 'decode', arrays)
    for m, shape, factor in [('eeg', (1, 6000, 1), 2.), ('hb', (1, 300, 2), 3.)]:
        assert decoded[m + '_sum'].shape == shape
        np.testing.assert_array_equal(decoded[m + '_S'] + decoded[m + '_O'], decoded[m + '_sum'])
        assert decoded[m + '_sum'][0, 0, 0] == 3 * factor
    high_id = np.full((1, 15), 2, dtype=np.uint8)
    assert apply(manifest, 'decode', dict(arrays, hb_O=high_id))['hb_O'][0, 0, 0] == 12.
    with pytest.raises(ValueError, match='code indices'):
        apply(manifest, 'decode', dict(arrays, hb_S=high_id))
    with pytest.raises(ValueError, match='exactly'):
        apply(manifest, 'decode', dict(arrays, window_scale=np.array([1.])))
    with pytest.raises(ValueError, match='code indices'):
        apply(manifest, 'decode', dict(arrays, eeg_S=arrays['eeg_S'].astype(float)))


def test_spatial_montage_order_scale_and_missing_observation():
    from experiments.scripts.apply_joint_tokenizer import spatial_eeg

    class Model:
        metadata = {'positions': {'study': ['A', 'B']}}

        @staticmethod
        def encode(values, dataset, observed):
            assert dataset == 'study'
            np.testing.assert_array_equal(observed, [[True, False]])
            np.testing.assert_array_equal(values[0, :, 0, 0], [2., 0.])
            return {b: np.tile(np.array([2, 0])[None, :, None], (1, 1, 15)) for b in ('S', 'O')}

        @staticmethod
        def decode_ids(ids, dataset):
            np.testing.assert_array_equal(ids['S'][0, :, 0], [2, 0])
            wave = np.broadcast_to(ids['S'][..., None], (1, 2, 15, 400))
            return {'S': wave, 'O': wave, 'sum': 2 * wave}

    scales = {'study': {'A': {'scale': 3.}, 'B': {'scale': 4.}}}
    data = dict(eeg_x=np.stack([np.full((6000, 1), np.nan), np.full((6000, 1), 6.)]),
                eeg_dataset=np.array(['study', 'study']), eeg_position=np.array(['B', 'A']),
                eeg_crop=np.array([0, 0]), eeg_observed=np.array([False, True]))
    ids = spatial_eeg(Model(), data, 'encode', scales)
    assert 'eeg_observed' not in ids
    native = spatial_eeg(Model(), ids, 'decode', scales)
    np.testing.assert_array_equal(native['eeg_sum'][:, 0, 0], [0., 12.])
    with pytest.raises(ValueError, match='exactly once'):
        spatial_eeg(Model(), dict(data, eeg_position=np.array(['A', 'A'])), 'encode', scales)
