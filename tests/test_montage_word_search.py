import numpy as np
from src.tokenizers.montage_word_search import synthesis_matrix, project, select_words
from src.tokenizers.cross_branch_innovations import posterior_waveforms,pilot_positions


def test_unfiltered_columns_compose_the_frozen_full_context_operator():
    rng=np.random.default_rng(197);n=4
    a=rng.normal(size=(5,2*n,2*n));cov=a@a.swapaxes(-1,-2)+np.eye(2*n)
    shape=(5,2*n);g=rng.uniform(.6,1.1,shape);r=np.ones(shape)*.2;scale=rng.uniform(.3,2.,shape)
    state=dict(joint=cov,pilot_gain=g,pilot_noise=r,innovation_mean_gain=g*.8,innovation_mean_noise=r*1.2,scale=scale)
    role=np.arange(2*n)%3;pilot=pilot_positions(n);values=rng.normal(size=(2*n,15,400))
    actual=project((synthesis_matrix(role,state)@values.reshape(2*n,6000)).reshape(3*n,15,400))
    previous=posterior_waveforms(values*pilot[:,None,None],values*~pilot[:,None,None]*scale[role,np.arange(2*n),None,None],role,cov,g,r,g*.8,r*1.2,'innovation')
    np.testing.assert_allclose(actual,np.concatenate([previous[b] for b in ['S','O','sum']]),rtol=1e-12,atol=1e-12)


def test_coordinate_search_can_recover_active_atom_without_extra_integers():
    a=np.sin(np.arange(400)*2*np.pi/25);atoms=np.array([0*a,a])
    base=dict(atoms=atoms,fast=atoms.T.astype(np.float32),norm=(atoms**2).sum(1),property_centers=np.zeros((2,3)),mean=0,scale=1,count=np.ones(2))
    table={('S',0,True):dict(base,offset=1),('O',0,False):dict(base,offset=5116)}
    sent={'S':np.ones((1,15),np.uint16),'O':np.full((1,15),5116,np.uint16)}
    m=np.array([[1,0],[0,1],[1,1]]);target=np.broadcast_to(a,(3,15,400)).copy();target[1]=0
    result,audit=select_words(sent,m,target,np.ones(3),{b:np.zeros((1,15,3)) for b in 'SO'},table,np.array([0,0]),2,3)
    assert np.all(result['S']==2) and np.all(result['O']==5116)
    assert audit['loss'][-1]<1e-10 and np.all(np.diff(audit['loss'])<=0)


def test_local_spectrum_adjoint_and_magnitude_majorization():
    from src.tokenizers.montage_word_search import local_spectrum, spectrum_adjoint, spectral_residual
    rng=np.random.default_rng(83)
    target=rng.normal(size=(3,15,400));base=project(rng.normal(size=target.shape));proposal=target*.15
    np.testing.assert_allclose(spectrum_adjoint(local_spectrum(target),target.shape),target,atol=4e-15)
    mag=abs(local_spectrum(target));direction,loss=spectral_residual(target,base,mag,2.)
    _,actual=spectral_residual(target,base+project(proposal),mag,2.)
    bound=loss+5*(np.square(proposal)-2*direction*proposal).sum((1,2))
    assert np.all(actual<=bound+1e-10)


def test_real_training_occupancy_excludes_empty_atoms_in_both_search_modes():
    wave=np.sin(np.arange(400)*2*np.pi/25);atoms=np.array([wave,0*wave,-wave])
    # The exact best waveform is a real empty cluster; only the occupied zero
    # and negative atoms can be proposed, irrespective of the semantic shortlist.
    book=dict(atoms=atoms,fast=atoms.T.astype(np.float32),norm=(atoms*atoms).sum(1),property_centers=np.zeros((3,3)),mean=0,scale=1,count=np.array([0,8,9]))
    table={('S',0,True):dict(book,offset=1),('O',0,False):dict(book,offset=5116)}
    sent=dict(S=np.full((1,15),3,np.uint16),O=np.full((1,15),5117,np.uint16));matrix=np.array([[1,0],[0,1],[1,1]])
    target=np.stack([np.broadcast_to(wave,(15,400)),np.zeros((15,400)),np.broadcast_to(wave,(15,400))])
    for limit in [0,64]:
        result,_=select_words(sent,matrix,target,np.ones(3),{b:np.zeros((1,15,3)) for b in 'SO'},table,np.array([0,0]),limit,3)
        assert np.all(result['S']!=1) and np.all(result['O']!=5116)


def test_portable_search_factory_requires_real_occupancy_variance_and_frozen_variant(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import pytest
    from experiments.scripts.apply_joint_tokenizer import load_branch_pair
    import src.tokenizers.montage_word_search_inference as module
    atom=np.zeros((2,400));centers=np.zeros((2,3));common=dict(property_mean=np.zeros(3),property_scale=np.ones(3))
    pilot=dict(common,balanced_atoms=atom,balanced_gain=np.ones(2),balanced_centers=centers,balanced_count=np.array([8,0]))
    residual=dict(common,innovation_atoms=atom,innovation_centers=centers,innovation_count=np.array([0,9]),capacity=2047)
    metadata=dict(encoder_extension='montage_word_search_v1',positions={'fixture':['a','b']},search_variants={'waveform':dict(limit=0,sweeps=3,spectral_strength=0.),'semantic64':dict(limit=64,sweeps=3,spectral_strength=0.)})
    base=SimpleNamespace(metadata=metadata,sources={},pilot_books={b:[pilot.copy() for _ in range(5)] for b in 'SO'},residual_books={b:[residual.copy() for _ in range(5)] for b in 'SO'},states={})
    monkeypatch.setattr(module,'load_innovation',lambda _:base)
    path=tmp_path/'model.npz';np.savez(path,fixture_train_target_variance=np.ones((3,2)))
    for name,limit in [('waveform',0),('semantic64',64)]:
        model=load_branch_pair('montage_word_search_eeg',path,name);assert model.selection['limit']==limit
        np.testing.assert_array_equal(model.word_tables['S',0,True]['count'],[8,0])
    # Decoding succeeds with every continuous source deliberately unavailable.
    n=4;state=dict(joint=np.broadcast_to(np.eye(n),(5,n,n)).copy(),scale=np.ones((5,n)))
    state.update({k:np.ones((5,n))*(.1 if 'noise' in k else 1.) for k in ['pilot_gain','pilot_noise','innovation_mean_gain','innovation_mean_noise']})
    model.states={'fixture':state};model.sources={'fixture':None}
    model._encode_array=lambda *_:(_ for _ in ()).throw(AssertionError('decode requested X'))
    ids=dict(S=np.array([[1]*15,[5116]*15])[None],O=np.array([[5116]*15,[1]*15])[None])
    decoded=model.decode_ids(ids,'fixture')
    for b in ['S','O','sum']:np.testing.assert_array_equal(decoded[b],np.zeros((1,2,15,400)))
    with pytest.raises(ValueError,match='registered'):load_branch_pair('montage_word_search_eeg',path,'unregistered')
    np.savez(path,fixture_train_target_variance=np.zeros((3,2)))
    with pytest.raises(ValueError,match='variances'):load_branch_pair('montage_word_search_eeg',path,'waveform')
    np.savez(path,fixture_train_target_variance=np.ones((3,2)));base.pilot_books['S'][0]['balanced_count']=np.zeros(2,int)
    with pytest.raises(ValueError,match='occupancy'):load_branch_pair('montage_word_search_eeg',path,'waveform')
