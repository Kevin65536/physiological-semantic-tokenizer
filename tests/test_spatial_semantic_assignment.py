import numpy as np

from src.tokenizers.spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, joint_assignment_features, load_model


def test_semantic_phase_code_preserves_whole_patch_information_and_integer_only_replay(tmp_path):
    time = np.arange(400) / 200
    low = np.sin(2*np.pi*2*time)
    high = low + np.sin(2*np.pi*10*time)
    atoms = {}
    centers = {}
    weights = {}
    distances = {}
    for b in ('S','O'):
        for k in range(len(BANDS)):
            vals = np.stack([low, high])
            data = joint_assignment_features(vals, k)
            distances[b,k] = np.ones(data.shape[-1])
            centers[b,k] = data
            atoms[b,k] = np.vstack([np.zeros(400), vals])
            weights['fixture',b,k] = np.ones((1,5))
    meta=dict(schema='spatial_waveform_dictionary_v1',codebook_size=3,
              positions={'fixture':list('ABCDE')},semantic_distance_fraction=.25)
    model=SpatialWaveformDictionary(meta,{b:np.ones(3001) for b in ('S','O')},centers,atoms,weights,feature_weights=distances)
    path=tmp_path/'model.npz'
    model.save(path)
    restored=load_model(path)
    x=np.tile(high,(1,5,15,1))
    ids=restored.encode(x,'fixture',np.ones((1,5),bool))
    np.testing.assert_array_equal(ids['S'],np.full((1,5,15),2))
    for b in ('S','O','sum'):
        np.testing.assert_array_equal(model.decode_ids(ids,'fixture')[b],restored.decode_ids(ids,'fixture')[b])
