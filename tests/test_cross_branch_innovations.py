"""Paid-pilot conditioning matches the direct full joint Gaussian update."""
import numpy as np
from src.tokenizers.cross_branch_innovations import condition_on_pilots,pilot_means,pilot_positions,posterior_waveforms
from src.tokenizers.cross_branch_sampling import reconstruct


def test_sequential_conditioning_equals_joint_direct_measurements():
    rng=np.random.default_rng(197);n=4
    matrix=rng.normal(size=(5,2*n,2*n));cov=matrix@matrix.swapaxes(-1,-2)+np.eye(2*n)
    roles=np.arange(2*n)%5
    wave=rng.normal(size=(2*n,15,400));pilot=pilot_positions(n)
    gain=rng.uniform(.6,1.4,size=(5,2*n));noise=rng.uniform(.1,.3,size=(5,2*n))
    first=np.where(pilot[:,None,None],wave,0.)
    second=np.where(~pilot[:,None,None],wave,0.)
    actual=posterior_waveforms(first,second,roles,cov,gain,noise,gain,noise,"direct")
    expected=reconstruct(wave,roles,cov,gain,noise)
    for b in actual:np.testing.assert_allclose(actual[b],expected[b],rtol=1e-11,atol=1e-11)


def test_conditional_innovation_restores_same_direct_result_without_side_inputs():
    rng=np.random.default_rng(198);n=3
    m=rng.normal(size=(5,2*n,2*n));cov=m@m.swapaxes(-1,-2)+np.eye(2*n)
    roles=np.arange(2*n)%5;pilot=pilot_positions(n)
    wave=rng.normal(size=(2*n,15,400));g=np.ones((5,2*n));noise=np.zeros_like(g)
    first=np.where(pilot[:,None,None],wave,0.);direct=np.where(~pilot[:,None,None],wave,0.)
    means,_=pilot_means(first,roles,cov,g,noise);innovation=direct.copy()
    for band in range(5):
        sites=np.flatnonzero((roles==band)&~pilot);innovation[sites]-=means[band,sites]
    actual=posterior_waveforms(first,innovation,roles,cov,g,noise,g,noise,"innovation")
    expected=posterior_waveforms(first,direct,roles,cov,g,noise,g,noise,"direct")
    for b in actual:np.testing.assert_allclose(actual[b],expected[b],rtol=1e-11,atol=1e-11)


def test_paid_words_include_type_and_fixed_scale_at_decoder():
    import pytest
    from src.tokenizers.cross_branch_innovations import paid_fields
    old={b:[dict(balanced_atoms=np.ones((1,400),np.float32),balanced_gain=np.ones(1)) for _ in range(5)] for b in "SO"}
    new={b:[dict(direct_atoms=np.ones((1,400),np.float32)*3,direct_energy_gain=np.ones(1)) for _ in range(5)] for b in "SO"}
    words={"S":np.broadcast_to(np.array([1,5116],np.uint16)[:,None],(2,15)).copy(),
           "O":np.broadcast_to(np.array([5116,1],np.uint16)[:,None],(2,15)).copy()}
    first,second,roles=paid_fields(words,old,new,np.ones((5,4))*2,"direct",False)
    np.testing.assert_array_equal(first[:,0,0],[1,0,0,1])
    np.testing.assert_array_equal(second[:,0,0],[0,6,6,0])
    np.testing.assert_array_equal(roles,[0,0,0,0])
    invalid={k:v.copy() for k,v in words.items()};invalid["S"][0]=5116
    with pytest.raises(ValueError):paid_fields(invalid,old,new,np.ones((5,4)),"direct",False)
    invalid={k:v.copy() for k,v in words.items()};invalid["S"][0]=0
    with pytest.raises(ValueError):paid_fields(invalid,old,new,np.ones((5,4)),"direct",False)


def test_residual_word_stride_comes_from_frozen_dictionary():
    from src.tokenizers.cross_branch_innovations import PILOT_SIZE, paid_fields
    pilot = {b:[dict(balanced_atoms=np.ones((1,400)),balanced_gain=np.ones(1)) for _ in range(5)] for b in 'SO'}
    residual = {b:[dict(capacity=2047,innovation_atoms=np.full((1,400),j+1.)) for j in range(5)] for b in 'SO'}
    word = PILOT_SIZE+4*2047
    ids = {'S':np.tile(np.array([[1],[word]]),(1,15)), 'O':np.tile(np.array([[word],[1]]),(1,15))}
    _, second, role = paid_fields(ids,pilot,residual,np.ones((5,4)),'innovation',False)
    np.testing.assert_array_equal(role,[0,4,4,0])
    np.testing.assert_array_equal(second[:,0,0],[0,5,5,0])
