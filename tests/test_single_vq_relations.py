import numpy as np
import pytest

from src.analysis.single_vq_relations import descriptors, profile, surrogate_batch, wave_attribute


def test_signal_coordinates_and_family_freeze():
    t=np.arange(400)/200
    assert wave_attribute(np.sin(2*np.pi*10*t),'eeg') > .999
    assert wave_attribute(np.sin(2*np.pi*5*t),'eeg') < .001
    hb=np.column_stack([2*np.arange(20)/10,-np.arange(20)/10])
    np.testing.assert_allclose(wave_attribute(hb.ravel(),'hb'),2)
    groups,n,means,_=profile(np.repeat([0,1,2],4),np.repeat([0.,1.,2.],4),4)
    assert n[3]==0 and means[3]==1 and groups[2]>=groups[0]


def test_exact_surrogate_constraints_and_duration():
    x=np.array([0,0,1,1,0,2,0,3,3,2,1,0,3,0])
    rng=np.random.default_rng(170)
    orig=descriptors(x[None])
    for _ in range(12):
        for mode in ('bag','transition','dwell'):
            y=surrogate_batch(x[None],('bag','transition','dwell').index(mode),int(rng.integers(100000)))[0]
            stats=descriptors(y[None])
            np.testing.assert_array_equal(orig[0],stats[0])
            if mode!='bag':
                assert (y[0],y[-1])==(x[0],x[-1])
                np.testing.assert_array_equal(orig[1][:,:20],stats[1][:,:20])
            if mode=='dwell':
                np.testing.assert_array_equal(orig[1],stats[1])


def test_disjoint_event_history_and_same_phase_controls(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    import single_vq_event_coupling as screen
    x=np.arange(130*200*4,dtype='float32').reshape(-1,4)
    real=screen.context(x,80,-26,2)
    shifted=screen.context(x,80,-69,2)
    assert real.shape==shifted.shape==(2,15,400) and shifted.max()<real.min()
    xx=x.copy();xx[55*200,0]=np.nan
    assert screen.context(xx,80,-26,2) is None
    t=np.arange(10)*28.
    meta=pd.DataFrame([dict(split=role,subject=f'{role}{sub}',record='session_01',condition=cls,event=j,
                            eeg_marker=p) for role,n in [('train',6),('validation',4)] for sub in range(n)
                       for j,p in enumerate(t) for cls in [('MA','BL')[j%2]]])
    idx=screen.donor_indices(meta)
    assert (idx>=0).all()
    np.testing.assert_array_equal(meta[['subject','record','condition']],meta.iloc[idx][['subject','record','condition']])
    assert np.all(abs(meta.eeg_marker.to_numpy()-meta.iloc[idx].eeg_marker.to_numpy())>30)
    rng=np.random.default_rng(91);n=len(meta)
    base=rng.normal(size=(n,4)); token=rng.normal(size=(n,8))
    y=token[:,:2]+rng.normal(scale=.005,size=(n,2))
    monkeypatch.setattr(screen.suite,'OUT',tmp_path)
    result=screen.coupling(meta,dict(base=base,aligned=token,continuous=token,shift=token[::-1],y=np.stack([y]*3,axis=1)))
    assert result['scored_people']==4 and result['selected_lag_s']==2
    table=pd.read_csv(tmp_path/'coupling_contrasts.csv').set_index(['color','reference'])
    assert table.loc[('two_color','base'),'difference']>.5
