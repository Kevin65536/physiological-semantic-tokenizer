import numpy as np
import pytest
from src.tokenizers.hb_local_eeg_source import fixed_neighborhoods,power_bank,pool_power,lags,allocate,fit_ridge


def test_geometry_is_fixed_train_only_and_rejects_personal_coordinates():
    pair=dict(id='p',channel_sets={'nearest6':list(range(6))},channel_set_names={'nearest6':list('abcdef')},coordinate_system='template',coordinate_units='normalized_head_unit')
    record=dict(dataset='d',split='train',eeg_channel_names=list('abcdef'),pairs=[pair],geometry_identity={'eeg':[{'measured_subject_coordinate':False}]})
    result=fixed_neighborhoods([dict(record,split='test',pairs=[]),record],{'d':['p']})
    assert result['nearest6']['d']['p']==list('abcdef')
    record['geometry_identity']['eeg'][0]['measured_subject_coordinate']=True
    with pytest.raises(ValueError):fixed_neighborhoods([record],{'d':['p']})


def test_native_local_band_feature_and_no_outside_lag():
    t=np.arange(6000)/200;x=np.stack([np.sin(2*np.pi*6*t),2*np.sin(2*np.pi*10*t)],1);observed=np.array([True,True]);p=power_bank(x,observed)
    a,_=pool_power(p,observed,[0]);b,_=pool_power(p,observed,[1]);assert a[:,1].mean()>.3 and b[:,2].mean()>1
    assert a[:,6].mean()>.99 and b[:,7].mean()>.99
    lag,mask=lags(a,np.ones(300,bool),[-3,3]);np.testing.assert_array_equal(lag[:30,10:],0);np.testing.assert_array_equal(lag[-30:,:10],0)
    assert mask[:30,1].sum()==0 and mask[-30:,0].sum()==0


def test_exact_fixed_sum_quiet_and_train_normalized_ridge():
    rng=np.random.default_rng(4);s=rng.normal(size=(2,15,40));o=rng.normal(size=s.shape);base=dict(S=s,O=o,sum=s+o);result=allocate(base,rng.normal(size=s.shape),np.array([True,False]))
    np.testing.assert_array_equal(result['sum'],base['sum'])
    for b in base:np.testing.assert_array_equal(result[b][0],base[b][0])
    np.testing.assert_allclose(result['S']+result['O'],base['sum'],atol=1e-15)
    x=rng.normal(size=(500,3));y=x@np.array([1.,2.,3.]);coef,_=fit_ridge(x.T@x,x.T@y,500,1e-10);np.testing.assert_allclose(coef,[1,2,3],rtol=1e-8)


def test_runner_rejects_nontrain_before_data_access():
    from pathlib import Path
    import yaml
    from experiments.scripts.diagnose_hb_local_eeg_source import validate
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_local_eeg_v1.yaml').read_text());validate(cfg)
    cfg['data']['allowed_splits'].append('validation')
    with pytest.raises(ValueError):validate(cfg)
