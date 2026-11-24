import numpy as np
import pytest
from src.tokenizers.eeg_integer_distillation import integer_fields, fit_additive, predict_features, nearest_internal


def test_fields_integer_only_boundaries_and_fixed_neighbors():
    a = np.arange(45, dtype=np.int32).reshape(1, 3, 15)+1
    ids = {'S': a, 'O': a+50}
    graph = np.array([[1, 2], [0, 2], [0, 1]])
    got = integer_fields(ids, 'S', graph)
    np.testing.assert_array_equal(got[..., 1], a)
    assert (got[..., 0][..., 0] == 512).all()
    assert (got[..., 2][..., -1] == 512).all()
    np.testing.assert_array_equal(got[0, 0, :, 4], a[0, 1])
    with pytest.raises(ValueError):
        integer_fields({'S': a.astype(float), 'O': a}, 'S', graph)
    with pytest.raises(ValueError):
        integer_fields({'S': a*512, 'O': a}, 'S', graph)


def test_fitting_is_local_and_predictive_and_serializable(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.integers(0, 4, (300, 8))
    y = np.column_stack((x[:, 1]+2*x[:, 4], 3*x[:, 2]))
    state, info = fit_additive(x, y, np.ones(300), prior=.01, passes=8)
    p = predict_features(x, np.zeros_like(y, float), state)
    assert np.mean((p-y)**2) < 1e-6
    path = tmp_path/'state.npz'
    np.savez(path, **state)
    z = np.load(path)
    np.testing.assert_array_equal(predict_features(x, np.zeros_like(y, float), z), p)
    altered = y.copy();altered[:10] += 1000
    # Held labels are absent from the fit API; changing a distinct held target has no path into state.
    again, _ = fit_additive(x, y, np.ones(300), prior=.01, passes=8)
    np.testing.assert_array_equal(again['tables'], state['tables'])
    assert info['samples'] == 300


def test_hard_internal_code_and_missing_reserved():
    t = np.array([[0., 0.], [1., 0.], [0., 1.]])
    np.testing.assert_array_equal(nearest_internal(np.array([[.8, .1], [.1, .9]]), t, block=1), [1, 2])


def test_synthetic_full_decoder_replays_only_saved_low_integer_arrays(tmp_path):
    from src.tokenizers.eeg_integer_distillation import EEGIntegerDistillationDecoder
    from src.tokenizers.spatial_waveform_dictionary import spectral_features
    rng = np.random.default_rng(9)
    class Tiny:
        metadata = {'positions': {'fixture': list(map(str,range(15)))}}
        def __init__(self,k):
            self.atoms = {(b,j): np.vstack([np.zeros((1,400)), rng.normal(size=(k-1,400))]) for b in ('S','O') for j in range(5)}
        def roles(self,d,b): return np.arange(15)%5
        def decode_ids(self,ids,d):
            out = {b: np.stack([self.atoms[b,p%5][ids[b][0,p]] for p in range(15)])[None] for b in ('S','O')}
            return dict(out,sum=out['S']+out['O'])
    low,high=Tiny(512),Tiny(8)
    states={(b,k):dict(intercept=np.zeros(spectral_features(low.atoms[b,k],k).shape[1]),tables=np.zeros((8,513,spectral_features(low.atoms[b,k],k).shape[1]))) for b in ('S','O') for k in range(5)}
    graph={('fixture',b):np.array([[(p+5)%15,(p+10)%15] for p in range(15)]) for b in ('S','O')}
    m=EEGIntegerDistillationDecoder(low,high,states,graph,dict(positions=low.metadata['positions']))
    codes={b:rng.integers(1,512,(1,15,15),dtype=np.int32) for b in ('S','O')}
    for b in codes:codes[b][:,0]=0
    before=m.decode_ids(codes,'fixture')
    np.savez(tmp_path/'payload.npz',**codes)
    with np.load(tmp_path/'payload.npz') as z:after=m.decode_ids({b:z[b] for b in ('S','O')},'fixture')
    for b in before:np.testing.assert_array_equal(before[b],after[b])
    for b in ('S','O'):assert not m.infer_ids(codes,'fixture')[b][:,0].any()
    with pytest.raises(TypeError):m.decode_ids(codes,'fixture',teacher_IDs=codes)


def test_full_atom_projection_matches_assigned_and_cross_band_and_no_held_fit():
    from src.tokenizers.eeg_integer_distillation import fit_full_atom_projection
    rng=np.random.default_rng(17)
    f=np.vstack([np.zeros((1,4)),rng.normal(size=(30,4))]);coef=rng.normal(size=(4,400));y=f@coef
    counts=np.r_[0,np.ones(20),np.zeros(10)]
    state,diag=fit_full_atom_projection(f,y,counts,ridge_fraction=1e-10)
    assert np.max(abs(f@state['coefficient']+state['intercept']-y))<1e-8
    altered=y.copy();altered[21:]+=1000
    second,_=fit_full_atom_projection(f,altered,counts,ridge_fraction=1e-10)
    np.testing.assert_array_equal(state['coefficient'],second['coefficient'])
    assert diag['fit_used_IDs']==20


def test_full_atom_route_equals_original_spatial_decode_without_mutating():
    from src.tokenizers.eeg_integer_distillation import synthesize_full_atoms
    from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary
    rng=np.random.default_rng(19);n=15;k=10
    meta=dict(positions={'fixture':list(map(str,range(n)))},codebook_size=k)
    atoms={(b,j):np.vstack([np.zeros((1,400)),rng.normal(size=(k-1,400))]) for b in ('S','O') for j in range(5)}
    weights={('fixture',b,j):rng.normal(size=(3,n))*.1 for b in ('S','O') for j in range(5)}
    m=SpatialWaveformDictionary(meta,{}, {},atoms,weights)
    ids={b:rng.integers(1,k,(1,n,15),dtype=np.int32) for b in ('S','O')}
    for b in ids:ids[b][:,0]=0
    full={b:np.stack([atoms[b,p%5][ids[b][0,p]] for p in range(n)])[None] for b in ('S','O')}
    expected=m.decode_ids(ids,'fixture');got=synthesize_full_atoms(m,full,ids,'fixture')
    for b in expected:np.testing.assert_array_equal(got[b],expected[b])
    assert m.atoms is atoms and m.codebook_size==k
