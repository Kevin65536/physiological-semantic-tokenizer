"""Complete atom support, role namespaces, and ID-only energy calibration."""
import numpy as np

from src.tokenizers.spatial_semantic_dictionary import SpatialSemanticDictionary, load_model, semantic_slot_features
from src.tokenizers.spectral_waveform_dictionary import band_components
from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary


def fixture_model():
    t = np.arange(400)/200
    prototypes = np.stack((np.sin(2*np.pi*5*t), np.sin(2*np.pi*10*t)))
    source = SupervisedWaveformDictionary(
        dict(schema='supervised_waveform_dictionary_v1', modality='eeg', codebook_size=2),
        {branch: np.full((3001, 1, 1), .5, complex) for branch in ('S', 'O')},
        {branch: np.zeros(400) for branch in ('S', 'O')},
        {branch: np.eye(400) for branch in ('S', 'O')},
        {branch: prototypes.copy() for branch in ('S', 'O')})
    metadata = dict(schema='spatial_semantic_dictionary_v1', positions={'fixture': ['A', 'B', 'C', 'D']},
                    slot_samples=100, codebook_size=3, interface_eligible=True, calibration_gain_cap=4)
    books = {}
    for branch in ('S', 'O'):
        for slot in range(4):
            books[branch, slot] = dict(feature_mean=np.zeros(103), feature_scale=np.ones(103),
                centers=semantic_slot_features(prototypes, slot), atoms=np.vstack((np.zeros(400), prototypes)),
                band_energy=np.vstack((np.zeros(5), np.mean(band_components(prototypes)**2, axis=-1))),
                rms_envelope=np.vstack((np.zeros(8), np.sqrt(np.mean(prototypes.reshape(2, 8, 50)**2, axis=-1)))),
                counts=np.array([0, 20, 20]))
    weights = {('fixture', mode, branch, slot): np.ones((1, 4)) for mode in ('continuous', 'quantized')
               for branch in ('S', 'O') for slot in range(4)}
    return SpatialSemanticDictionary(metadata, source, books, weights), prototypes


def test_whole_two_second_attributes_affect_role_features():
    wave = np.zeros((2, 400)); wave[1, 200:] = np.sin(2*np.pi*10*np.arange(200)/200)
    features = semantic_slot_features(wave, 0)
    np.testing.assert_array_equal(features[0, 3:], features[1, 3:])
    assert abs(features[0, 1]-features[1, 1]) > 1


def test_full_atom_ids_decode_and_calibrate_after_model_reload(tmp_path):
    model, atoms = fixture_model()
    x = np.tile((2*atoms[0])[None, None, None, :], (1, 4, 15, 1))
    ids = model.encode(x, 'fixture', np.ones((1, 4), bool))
    for branch in ('S', 'O'):
        np.testing.assert_array_equal(ids[branch], np.ones((1, 4, 15), np.int32))
    uncalibrated = model.decode_ids(ids, 'fixture', calibrated=False)
    calibrated = model.decode_ids(ids, 'fixture', calibrated=True)
    np.testing.assert_allclose(uncalibrated['S'], np.tile(atoms[0][None, None, None], (1, 4, 15, 1)), atol=1e-14)
    path = tmp_path/'model.npz'; model.save(path)
    restored = load_model(path)
    restored.source.predict_continuous = lambda *_: (_ for _ in ()).throw(AssertionError('decoder accessed X predictor'))
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(calibrated[branch], restored.decode_ids(ids, 'fixture')[branch])
    assert restored.books['S', 0]['atoms'].shape == (3, 400)


def test_missing_role_is_encoded_in_ids_and_keeps_finite_decoding():
    model, _ = fixture_model()
    ids = {branch: np.ones((1, 4, 15), np.int32) for branch in ('S', 'O')}
    for branch in ids:
        ids[branch][:, 2] = 0
    output = model.decode_ids(ids, 'fixture')
    assert all(np.isfinite(values).all() for values in output.values())
