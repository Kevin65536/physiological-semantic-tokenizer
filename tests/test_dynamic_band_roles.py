import numpy as np

from src.tokenizers.cross_branch_sampling import reconstruct
from src.tokenizers.dynamic_band_roles import continuous_geometry, select_single_role


def fixture():
    rng=np.random.default_rng(197)
    n=4
    a=rng.normal(size=(5,2*n,2*n))
    cov=a@a.swapaxes(-1,-2)+np.eye(2*n)
    fields=rng.normal(size=(5,2*n,15,400))
    sp=np.fft.rfft(fields.reshape(5,2*n,6000))
    hz=np.fft.rfftfreq(6000,1/200)
    ranges=((1,4),(4,8),(8,13),(13,30),(30,45.00001))
    for b,(lo,hi) in enumerate(ranges):sp[b,:,(hz<lo)|(hz>=hi)]=0
    fields=np.fft.irfft(sp,n=6000).reshape(fields.shape)
    return fields,np.arange(2*n)%5,cov


def test_continuous_diagnosis_matches_unit_gain_zero_noise_paid_observations():
    fields,role,cov=fixture()
    selected=fields[role,np.arange(len(role))]
    result=reconstruct(selected,role,cov,np.ones(role.shape[0])[None].repeat(5,axis=0),np.zeros((5,len(role))))
    _,actual=continuous_geometry(fields,role,cov)
    np.testing.assert_allclose(actual,np.concatenate([result[b] for b in ('S','O','sum')]),rtol=1e-11,atol=1e-11)


def test_one_site_search_keeps_baseline_and_recovers_missing_strong_band():
    n=2
    cov=np.repeat(np.eye(2*n)[None],5,axis=0)
    fields=np.zeros((5,2*n,15,400))
    fields[4,2]=np.sin(np.arange(400)*2*np.pi*35/200)
    role=np.zeros(2*n,dtype=np.int8)
    target=np.zeros((3*n,15,400));target[2]=fields[4,2];target[4]=fields[4,2]
    fixed,dynamic,changed,audit=select_single_role(fields,role,cov,target,np.ones(3*n))
    assert np.count_nonzero(changed!=role)==1 and changed[2]==4
    assert audit['selected_branch']=='O' and audit['selected_native']==0
    assert audit['objective_after']<1e-20 and audit['objective_before']>0
    np.testing.assert_array_equal(dynamic,target)
