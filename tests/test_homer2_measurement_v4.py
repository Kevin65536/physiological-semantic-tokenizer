"""V4 uses official TDDR without changing historical measurement branches."""
import numpy as np
import pytest
import mne
from mne.preprocessing.nirs import temporal_derivative_distribution_repair
from src.data.homer2_preprocessing import (
    apply_homer2_aligned_contract, MEASUREMENT_ALIGNMENT_V4_SCHEMA,
    MEASUREMENT_ALIGNMENT_SCHEMA, HOMER2_ALIGNMENT_SCHEMA,
)


def apply(values, method='none', **kwargs):
    options=dict(dataset_id='eeg_fnirs_single_trial',sample_rate_hz=10.,entry_stage='optical_density',
        wavelengths_nm=(760.,850.),processing_schema=MEASUREMENT_ALIGNMENT_V4_SCHEMA,
        motion_method=method,retain_feature_boundary=True)
    options.update(kwargs)
    return apply_homer2_aligned_contract(values,**options)


def test_multiple_pairs_match_public_mne_and_legacy_bool_is_unused():
    t=np.arange(300)/10.
    od=np.stack([np.sin(t)*.01,np.cos(t)*.02,np.sin(t*.7)*.03,np.cos(t*.8)*.04],axis=1)
    info=mne.create_info(['S1_D1 760','S1_D1 850','S2_D2 760','S2_D2 850'],10.,['fnirs_od']*4)
    for i,ch in enumerate(info['chs']):
        ch['loc'][:3]=[.015,0,0];ch['loc'][3:6]=[0,0,0];ch['loc'][6:9]=[.03,0,0];ch['loc'][9]=[760.,850.][i%2]
    reference=temporal_derivative_distribution_repair(mne.io.RawArray(od.T,info,verbose=False),verbose=False).get_data().T
    result=apply(od.reshape(300,2,2),'mne_tddr')
    np.testing.assert_array_equal(result.pre_linear_optical_density.reshape(300,4),reference)
    other=apply(od.reshape(300,2,2),'mne_tddr',motion_correction=False)
    np.testing.assert_array_equal(result.values,other.values)
    assert result.state.parameters['motion_correction'] is True
    assert result.values.dtype==np.float64


def test_no_motion_preserves_od_and_true_linear_trend():
    t=np.arange(1201)/10.
    od=np.stack([.001*t,.002*t],axis=1)[:,None,:]
    result=apply(od)
    np.testing.assert_array_equal(result.pre_linear_optical_density,od)
    assert result.state.parameters['motion_correction'] is False
    repaired=apply(od,'mne_tddr').pre_linear_optical_density
    assert abs(repaired[-1,0,0]-repaired[0,0,0])<.1*(od[-1,0,0]-od[0,0,0])


@pytest.mark.parametrize('stage',['raw_intensity','optical_density'])
@pytest.mark.parametrize('method',['none','mne_tddr'])
def test_masked_values_cannot_affect_v4(stage,method):
    t=np.arange(300)/10.
    values=np.stack([.01*np.sin(t),.02*np.cos(t)],axis=1)[:,None,:]
    if stage=='raw_intensity':values=np.exp(-values)
    mask=np.ones_like(values,bool);mask[80:120]=False
    changed=values.copy();changed[~mask]=12345.
    first=apply(values,method,entry_stage=stage,valid_mask=mask)
    second=apply(changed,method,entry_stage=stage,valid_mask=mask)
    np.testing.assert_array_equal(first.pre_linear_optical_density,second.pre_linear_optical_density)
    np.testing.assert_array_equal(first.values,second.values)
    assert not first.recorded_mask[80:120].any()
    assert not first.processed_valid_mask.any()


@pytest.mark.parametrize('sign',[1.,-1.])
def test_motion_free_asymmetric_pulse_avoids_legacy_accumulated_drift(sign):
    t=np.arange(1201)/10.;phase=t%1
    pulse=.03*sign*np.where(phase<.2,.5-.5*np.cos(np.pi*phase/.2),.5+.5*np.cos(np.pi*(phase-.2)/.8))
    od=np.stack([pulse,.7*pulse],axis=1)[:,None,:]
    old=apply(od,method=None,processing_schema=MEASUREMENT_ALIGNMENT_SCHEMA).pre_linear_optical_density
    new=apply(od,'mne_tddr').pre_linear_optical_density
    none=apply(od).pre_linear_optical_density
    assert abs(old[-1,0,0]-old[0,0,0])>1.
    assert abs(new[-1,0,0]-new[0,0,0])<.01*abs(old[-1,0,0]-old[0,0,0])
    np.testing.assert_array_equal(none,od)


@pytest.mark.parametrize('kwargs',[
    {'motion_method':None},{'motion_method':'legacy'},{'entry_stage':'chromophore'},
    {'processing_schema':'unrecognized'},{'wavelengths_nm':(850.,760.)},
    {'sample_rate_hz':0.},{'sample_rate_hz':np.nan},
    {'processing_schema':MEASUREMENT_ALIGNMENT_SCHEMA},
    {'processing_schema':HOMER2_ALIGNMENT_SCHEMA},
])
def test_invalid_v4_or_legacy_method_rejected(kwargs):
    with pytest.raises(ValueError):apply(np.ones((30,1,2)),**kwargs)


@pytest.mark.parametrize('shape',[(30,2),(30,1,3),(30,0,2)])
def test_invalid_shape_rejected(shape):
    with pytest.raises(ValueError):apply(np.ones(shape),'mne_tddr')
