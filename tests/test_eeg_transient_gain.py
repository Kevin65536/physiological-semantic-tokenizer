import numpy as np
import pytest
from src.tokenizers.eeg_transient_gain import (
    prepare_x,feature_field,fit_background,fit_gain,apply_gain,BoundedTransientSource,load_model,band_indices,
)


def synthetic(seed):
    rng=np.random.default_rng(seed);t=np.arange(6000)/200
    clean=np.sin(2*np.pi*10*t[:,None]+rng.uniform(-np.pi,np.pi,(1,4)))
    true_peak=5*np.exp(-((t-18)/.22)**2)*np.sin(2*np.pi*10*t)
    clean[:,seed%4] += true_peak
    artifact=9*np.exp(-((t-(5+seed%5))/.23)**2)*np.sin(2*np.pi*10*t)
    x=clean+artifact[:,None]
    transfer={b:np.full(3001,.5) for b in 'SO'}
    p=prepare_x(x,np.ones(4,bool),transfer)
    spectrum=np.fft.rfft(clean.T,axis=-1);energy=[];cross=[]
    for b,idx in enumerate(band_indices()):
        f=np.zeros_like(spectrum);f[:,idx]=spectrum[:,idx]
        target=np.fft.irfft(f,n=6000,axis=-1)
        u=p['carriers'][b].reshape(4,300,20)
        energy.append((u*u).mean(-1));cross.append((u*target.reshape(4,300,20)).mean(-1))
    item=dict(prepared=p,frame_energy=np.array(energy),frame_cross=np.array(cross),weight=1/12)
    return item,x,clean


def test_bounded_X_transients_improve_synthetic_separation_without_changing_sum():
    train=[synthetic(i)[0] for i in range(12)]
    background=fit_background(train)
    beta=fit_gain(train,background,True);static=fit_gain(train,background,False)
    errors=[]
    for seed in range(12,16):
        item,x,truth=synthetic(seed);p=item['prepared']
        predicted,g=apply_gain(p,background,beta,True)
        control,_=apply_gain(p,background,static,False)
        assert g.min()>=0 and g.max()<=3
        np.testing.assert_allclose(predicted['S']+predicted['O'],p['prior']['sum'],rtol=0,atol=1e-12)
        errors.append([np.mean((p['prior']['S'].T-truth)**2),np.mean((control['S'].T-truth)**2),np.mean((predicted['S'].T-truth)**2)])
    error=np.mean(errors,axis=0)
    assert error[2]<.8*error[0]
    assert error[2]<.95*error[1]


def test_source_is_X_only_missing_safe_and_frozen_reload_exact(tmp_path):
    item,x,_=synthetic(7);background=fit_background([item]);beta=fit_gain([item],background)
    meta=dict(schema='eeg_bounded_transient_source_v1',positions={'fixture':['a','b','c','d']},transient=True,gain_bounds=[0,3],stage='continuous_only')
    model=BoundedTransientSource({b:np.full(3001,.5) for b in 'SO'},{'fixture':dict(background=background,beta=beta)},meta)
    seen=np.array([[True,True,True,False]]);values=x.T.reshape(1,4,15,400)
    original=model.predict_continuous(values,'fixture',seen)
    canary=values.copy();canary[:,3]=1e9
    for b,v in model.predict_continuous(canary,'fixture',seen).items():np.testing.assert_array_equal(v,original[b])
    model.save(tmp_path/'model.npz');replayed=load_model(tmp_path/'model.npz').predict_continuous(values,'fixture',seen)
    for b,v in replayed.items():np.testing.assert_array_equal(v,original[b])
    np.testing.assert_array_equal(original['S'][:,3],0)
    with pytest.raises(ValueError):model.predict_continuous(values[:,:,:14],'fixture',seen)


def test_ablated_features_are_constant_in_time_and_fields_saturate():
    item,x,_=synthetic(8);p=item['prepared'];bg=fit_background([item])
    f=feature_field(p,bg,2,False)
    np.testing.assert_array_equal(f[...,3:],0)
    np.testing.assert_array_equal(f,np.broadcast_to(f[:,:1],f.shape))
    full=feature_field(prepare_x(x*1e7,np.ones(4,bool),{b:np.full(3001,.5) for b in 'SO'}),bg,2,True)
    assert np.isfinite(full).all() and abs(full).max()<=1
