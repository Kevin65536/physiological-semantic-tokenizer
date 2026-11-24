import numpy as np
import pytest
from src.tokenizers.phase_orbit_dictionary import expansion, waveform_features, nearest, fit_book, PhaseOrbitDictionary, load_model
from src.tokenizers.spatial_waveform_dictionary import spectral_features, BANDS
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier


def test_expanded_shift_sign_atoms_pay_ids_and_match_full_waveform_distances():
    rng = np.random.default_rng(82)
    parents = spectral_features(rng.normal(size=(2, 400)), 1)
    shifts = list(range(0, 400, 25))
    atoms = waveform_features(expansion(parents, 1, shifts), 1)
    original = waveform_features(parents, 1)
    for p in range(2):
        for s, shift in enumerate(shifts):
            for sign in range(2):
                np.testing.assert_allclose(atoms[2*(16*p+s)+sign], (1-2*sign)*np.roll(original[p], shift), atol=2e-15)
    x = rng.normal(size=(10, 400))
    ids, _ = nearest(spectral_features(x, 1), expansion(parents, 1, shifts))
    np.testing.assert_array_equal(ids, ((x[:, None]-atoms[None])**2).sum(-1).argmin(-1))


def test_alignment_prevents_phase_mean_cancellation_without_amplitude_side_channel():
    rng = np.random.default_rng(91)
    parent = spectral_features(rng.normal(size=(1, 400)), 1)
    shifts = list(range(0, 400, 25))
    atoms = waveform_features(expansion(parent, 1, shifts), 1)
    carriers = np.tile(atoms, (4, 1))
    book, report = fit_book(carriers, 1, method='orbit', parent_count=1, shifts=shifts, iterations=3, seed=84)
    assert report['final_projected_MSE'] < 1e-12
    assert report['aligned_parent_mean_energy_fraction'] > .99999
    assert report['unaligned_same_parent_mean_energy_fraction'] < 1e-12
    assert book['atoms'].shape == (33, 400)


def test_saved_integer_replay_has_no_phase_or_continuous_inputs(tmp_path):
    rng = np.random.default_rng(12); books = {}
    for b in 'SO':
        for k in range(5):
            q = spectral_features(rng.normal(size=(1, 400)), k)
            center = expansion(q, k, list(range(16)))
            books[b, k] = dict(mean=np.zeros(center.shape[1]+3), weight=np.r_[np.ones(center.shape[1]), np.zeros(3)],
                centers=np.column_stack((center, np.zeros((32, 3)))), atoms=np.vstack((np.zeros((1, 400)), waveform_features(center,k))),
                counts=np.r_[0,np.ones(32)*8], iterations=1)
    a = {( 'fixture', b,k):np.eye(5)[:,k:k+1] for b in 'SO' for k in range(5)}
    s = {( 'fixture', b,k):np.eye(5)[k:k+1] for b in 'SO' for k in range(5)}
    source = AnchoredSpatialCarrier(dict(schema='anchored_spatial_carrier_v1',positions={'fixture':list('abcde')},bands_hz=BANDS),
        {b:np.full(3001,.5) for b in 'SO'},a,s)
    source_path=tmp_path/'source.npz'; source.save(source_path)
    model=PhaseOrbitDictionary(source,books,dict(schema='anchored_carrier_dictionary_v1',experiment_family='phase_orbit_dictionary_v1',codebook_size=33))
    x=rng.normal(size=(1,5,15,400)); observed=np.array([[True,True,False,True,True]])
    ids=model.encode(x,'fixture',observed); expected=model.decode_ids(ids,'fixture')
    model.save(tmp_path/'model.npz',source_path); np.savez(tmp_path/'ids.npz',**ids)
    restored=load_model(tmp_path/'model.npz')
    restored.source.continuous_source=lambda *_: (_ for _ in ()).throw(AssertionError('X requested'))
    restored.source.analyze=lambda *_: (_ for _ in ()).throw(AssertionError('carrier requested'))
    with np.load(tmp_path/'ids.npz') as disk: replay=restored.decode_ids(dict(disk),'fixture')
    for b in ('S','O','sum'):np.testing.assert_array_equal(expected[b],replay[b])
    for b in 'SO':np.testing.assert_array_equal(ids[b][:,2],0)
    with pytest.raises(ValueError,match='integer'):restored.decode_ids({b:v.astype(float) for b,v in ids.items()},'fixture')


def test_registered_vocabulary_and_boundaries_before_data():
    from pathlib import Path
    import yaml
    from experiments.scripts.run_phase_orbit_dictionary import validate
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_phase_orbit_v1.yaml').read_text())
    validate(cfg)
    assert cfg['model']['vocabulary_including_missing']==31*16*2+1
    cfg['data']['evaluation_splits']=['validation']
    with pytest.raises(ValueError,match='train-only'):validate(cfg)


def test_exact_10bit_transport_retains_every_expanded_ID_and_rejects_hidden_padding():
    from experiments.scripts.run_phase_orbit_dictionary import pack_integer_payload,unpack_integer_payload
    values=np.r_[np.arange(993),np.array([0,992,1],np.int32)]
    payload=pack_integer_payload(values)
    assert len(payload)==(len(values)*10+7)//8
    np.testing.assert_array_equal(unpack_integer_payload(payload,len(values)),values)
    odd=pack_integer_payload(np.array([992],np.int32))
    bad=bytes([odd[0],odd[1]|128])
    with pytest.raises(ValueError,match='padding'):unpack_integer_payload(bad,1)
    with pytest.raises(ValueError,match='integer'):pack_integer_payload(np.array([993],np.int32))
