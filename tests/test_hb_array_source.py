from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from src.tokenizers.hb_array_source import lagged_array,fit_transfer,predict_transfer,allocate,local_columns
from experiments.scripts.run_hb_array_source import validate


def test_time_support_and_native_masks_never_wrap():
    x=np.zeros((300,2,2));x[0,0,0]=3;x[-1,1,1]=7;seen=np.ones((300,2),bool);seen[50:60,1]=False
    a=lagged_array(x,seen);assert a.shape==(300,1+9*6)
    # Last registered positive8s lag cannot read the window tail at its start.
    offset=1+8*6
    assert np.all(a[:80,offset:offset+6]==0)
    assert a[80,offset]==3
    # First negative8s lag reads future; no artificial first impulse at the end.
    assert np.all(a[-80:,1:7]==0)
    assert len(local_columns(2,2,1))==28
    with pytest.raises(ValueError,match='300-time'):lagged_array(x,np.ones((300,3),bool))


def test_train_only_supervised_space_time_recovers_remote_delayed_component_and_reloads(tmp_path):
    rng=np.random.default_rng(17);n=2400;a=rng.normal(size=(n,18));y=np.c_[a[:,0]*2-a[:,15],a[:,2]-a[:,10]*3]
    fit=fit_transfer(a[:1800],y[:1800],np.ones((1800,2)));pred=predict_transfer(a[1800:],fit)
    assert fit['rank']==2 and np.mean((pred-y[1800:])**2)<.003
    altered=y.copy();altered[1800:]=999
    repeated=fit_transfer(a[:1800],altered[:1800],np.ones((1800,2)))
    np.testing.assert_array_equal(repeated['projection'],fit['projection'])
    np.savez(tmp_path/'fit.npz',**fit)
    with np.load(tmp_path/'fit.npz') as disk:restored={k:disk[k] for k in disk}
    np.testing.assert_array_equal(predict_transfer(a[1800:],restored),pred)
    masked=np.ones((1800,2));masked[:300]=0;y2=y[:1800].copy();y2[:300]=1e6
    f=fit_transfer(a[:1800],y2,masked);assert np.mean((predict_transfer(a[1800:],f)-y[1800:])**2)<.004


def test_local_control_excludes_other_native_channels_and_sum_quiet_exact():
    rng=np.random.default_rng(7);x=rng.normal(size=(300,2,2));a=lagged_array(x,np.ones((300,2),bool));y=np.c_[x[:,0],x[:,1]];w=np.ones_like(y)
    fitted=fit_transfer(a,y,w,local=(2,2));coeff=fitted['coefficient']
    allowed=set(local_columns(2,2,0));assert not np.any(coeff[[i for i in range(len(coeff)) if i not in allowed],:2])
    s=rng.normal(size=(3,15,40));o=rng.normal(size=s.shape);base=dict(S=s,O=o,sum=s+o);p=allocate(base,np.ones_like(s),[False,True,False])
    np.testing.assert_array_equal(p['sum'],base['sum'])
    for b in base:np.testing.assert_array_equal(p[b][1],base[b][1])
    np.testing.assert_allclose(p['S']+p['O'],p['sum'],atol=1e-14)


def test_scope_and_single_setting_before_measurements():
    p=Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_array_source_v1.yaml';c=yaml.safe_load(p.read_text());validate(c)
    for section,key,value in [('data','numerical_read_splits',['train','validation']),('data','teacher_refit',True),('model','ridge',.1),('model','lags_seconds',[0])]:
        changed=deepcopy(c);changed[section][key]=value
        with pytest.raises(ValueError,match='before arrays'):validate(changed)
