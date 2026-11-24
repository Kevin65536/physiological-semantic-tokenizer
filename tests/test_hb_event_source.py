from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
import pytest,yaml
from src.tokenizers.hb_event_source import event_features,centered,project_event,fit_event,predict_event,allocate_event
from experiments.scripts.run_hb_event_source import validate


def test_visible_local_and_array_features_exclude_hidden_positions_and_never_wrap():
    x=np.zeros((2,300,3,2));x[0,0,0,0]=3;x[0,-1,1,0]=9;seen=np.ones(x.shape[:3],bool);seen[:,:,2]=False
    base=np.zeros((1,300,4));local=event_features(x,seen,base,[0],[0],False);array=event_features(x,seen,base,[0],[0],True)
    changed=x.copy();changed[:,:,2]=1e9
    np.testing.assert_array_equal(event_features(changed,seen,base,[0],[0],True),array)
    changed[:,:,1]+=8
    np.testing.assert_array_equal(event_features(changed,seen,base,[0],[0],False),local)
    assert not np.array_equal(event_features(changed,seen,base,[0],[0],True),array)
    # Negative8s lag at last point holds endpoint, availability says outside.
    width=(local.shape[-1]-1)//9;assert local[0,-1,1+width-1]==0
    # Positive8s lag at t=0 holds first point, never reads tail.
    assert local[0,0,1+8*width]==3


def test_projection_is_declared_02Hz_no_DC_and_exact_fixed_sum_quiet():
    t=np.arange(300)+.5;wave=(3+np.cos(np.pi*t*4/300)+np.cos(np.pi*t*20/300))[None,:,None]*np.ones((2,1,2));expected=np.cos(np.pi*t*4/300)[None,:,None]*np.ones((2,1,2))
    np.testing.assert_allclose(project_event(wave),expected,atol=1e-14)
    s=np.ones((2,15,40));o=np.ones_like(s)*2;base=dict(S=s,O=o,sum=s+o);out=allocate_event(base,wave,[False,True]);np.testing.assert_array_equal(out['sum'],base['sum'])
    for b in base:np.testing.assert_array_equal(out[b][1],base[b][1])
    np.testing.assert_allclose(out['S']+out['O'],base['sum'],atol=1e-14);np.testing.assert_allclose((out['S']-s).reshape(2,300,2).mean(1),0,atol=1e-14)
    m=np.ones((2,300,2),bool);m[:,:10]=False;y=wave.copy();y[:,:10]=1e6
    np.testing.assert_allclose(np.where(m,centered(y,m),0).sum(1),0,atol=1e-10)


def test_nonlinear_train_fit_and_pickle_replay_respect_zero_weight_hidden_targets():
    rng=np.random.default_rng(4);a=rng.normal(size=(2400,6));a[:,0]=np.arange(len(a))%3;y=np.c_[a[:,1]*a[:,2],np.maximum(a[:,3],0)];w=np.ones_like(y);w[:100]=0
    fitted=fit_event(a[:2000],y[:2000],w[:2000],'nonlinear');restored=pickle.loads(pickle.dumps(fitted));p=predict_event(a[2000:],fitted)
    np.testing.assert_array_equal(p,predict_event(a[2000:],restored))
    poisoned=y[:2000].copy();poisoned[:100]=1e9;again=fit_event(a[:2000],poisoned,w[:2000],'nonlinear');np.testing.assert_array_equal(p,predict_event(a[2000:],again))
    assert np.mean((p-y[2000:])**2)<np.mean(y[2000:]**2)


def test_contract_rejects_boundary_or_retuning_before_arrays():
    c=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_event_source_v1.yaml').read_text());validate(c)
    for sec,key,value in [('boundary','numerical_read_splits',['validation']),('boundary','teacher_refit',True),('model','tree_iterations',240),('model','sample_stride',10)]:
        v=deepcopy(c);v[sec][key]=value
        with pytest.raises(ValueError,match='before arrays'):validate(v)
