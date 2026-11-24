import numpy as np
import pytest
from src.tokenizers.hb_component_search import objective,FrozenHb111


def test_objective_distinguishes_exact_sum_from_real_opposite_components():
    x=np.zeros((2,15,40));v=np.ones((2,1,40));p=dict(S=np.ones_like(x)*3,O=np.ones_like(x)*-3,sum=x)
    np.testing.assert_array_equal(objective(p,x,v),0)
    true=dict(S=np.ones_like(x),O=-np.ones_like(x))
    np.testing.assert_array_equal(objective(p,x,v,oracle=true,component_variance=dict(S=v,O=v)),8)
    source=dict(S=np.ones_like(x)*2,O=-np.ones_like(x)*2)
    np.testing.assert_array_equal(objective(p,x,v,source=source,component_variance=dict(S=v,O=v)),2)
    with pytest.raises(ValueError):objective(p,x[:,:14],v)


def test_reserved_pair_is_whole_context_only():
    ids=dict(S=np.zeros((2,15),int),O=np.zeros((2,15),int));ids['S'][0]=128;ids['O'][0]=256;ids['S'][1,0]=128
    assert FrozenHb111.quiet(ids).tolist()==[True,False]


def test_finite_search_accepts_only_real_words_and_keeps_reserved_context():
    class Fixture(FrozenHb111):
        def __init__(self):self.sizes=dict(S=2,O=2)
        def decode(self,ids,fold,gamma=1.):
            quiet=self.quiet(ids)
            s=np.repeat(np.asarray(ids['S'],float)[...,None],40,axis=2)
            o=np.repeat(np.asarray(ids['O'],float)[...,None],40,axis=2)
            s[quiet]=2;o[quiet]=-2
            return dict(S=s,O=o,sum=s+o)
    decoder=Fixture();ids=dict(S=np.zeros((2,15),np.uint16),O=np.zeros((2,15),np.uint16));ids['S'][1]=128;ids['O'][1]=256
    x=np.ones((2,15,40))*2;x[1]=0;v=np.ones((2,1,40))
    selected,report=decoder.search(ids,x,v,0,full=True)
    for b in 'SO':np.testing.assert_array_equal(selected[b][0],1);np.testing.assert_array_equal(selected[b][1],ids[b][1])
    assert report['changed']==30
    np.testing.assert_array_equal(objective(decoder.decode(selected,0),x,v),0)
    assert np.all(report['final_objective']<=report['initial_objective'])


def test_coupled_search_crosses_cancellation_barrier_with_exact_visible_guard():
    class Fixture(FrozenHb111):
        def __init__(self):self.sizes=dict(S=2,O=2)
        def decode(self,ids,fold,gamma=1.):
            s=np.repeat(np.asarray(ids['S'],float)[...,None],40,axis=2)
            o=-np.repeat(np.asarray(ids['O'],float)[...,None],40,axis=2)
            return dict(S=s,O=o,sum=s+o)
    decoder=Fixture();ids={b:np.zeros((1,15),np.uint16) for b in 'SO'};x=np.zeros((1,15,40));v=np.ones((1,1,40));source=dict(S=np.ones_like(x),O=-np.ones_like(x));cv=dict(S=v,O=v)
    single,_=decoder.search(ids,x,v,0,source,cv,full=True)
    np.testing.assert_array_equal(single['S'],0)
    paired,info=decoder.search_pairs(ids,x,v,0,source,cv,full=True)
    for b in 'SO':np.testing.assert_array_equal(paired[b],1)
    np.testing.assert_array_equal(decoder.decode(paired,0)['sum'],x)
    assert info['final_objective'][0]==0 and info['maximum_affine_proposal_gap']==0


def test_cartesian_quadratic_matches_explicit_both_color_errors():
    from src.tokenizers.hb_component_search import pair_quadratic,weighted_error
    rng=np.random.default_rng(84);e=rng.normal(size=(2,15,40));a=rng.normal(size=(2,3,15,40));b=rng.normal(size=(2,4,15,40));v=rng.uniform(.4,2,size=(2,1,40));mask=rng.uniform(size=e.shape)>.15
    for c in [None,0,1]:
        table=pair_quadratic(e,a,b,v,mask,c)
        expected=np.array([[weighted_error(e+a[:,i]+b[:,j],v,mask,c) for j in range(4)] for i in range(3)]).transpose(2,0,1)
        np.testing.assert_allclose(table,expected,rtol=1e-14,atol=1e-14)
