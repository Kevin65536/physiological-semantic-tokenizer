"""Reconstruction audits preserve native units, scoring masks, and failed cells."""
import numpy as np
import pytest

from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary


def batch_from_wave(y, channels):
    x = np.asarray(y).reshape(len(y), 15, -1)
    return dict(x=x, s=x*.6, o=x*.4, mask=np.ones_like(x, bool),
                **{key: np.ones_like(x)*value for key,value in [('vx',2.),('vs',.5),('vo',.25),
                                                              ('dvx',1.),('dvs',.5),('dvo',.25)]})


def test_eeg_gain_and_native_denominators_with_masked_nan_tail():
    t = np.arange(6000)/200
    y = ((1+.3*np.sin(2*np.pi*.2*t))*np.sin(2*np.pi*10*t))[None,:,None]
    b = batch_from_wave(y, 1)
    b['mask'][0,-1,-5:] = False
    for key in ['x','s','o']:
        b[key][~b['mask']] = np.nan
    p = dict(sum=2*b['x'],S=2*b['s'],O=2*b['o'])
    out = joint_reconstruction_metrics(p,b,1,[3.])
    m = out['metrics']
    assert m['nmse'].shape == (1,3,1)
    np.testing.assert_allclose(m['band_8_13_db'], 10*np.log10(4), atol=1e-12)
    np.testing.assert_allclose(m['local_rms_correlation'],1.,atol=1e-12)
    np.testing.assert_allclose(m['local_rms_nrmse'],1.,atol=1e-12)
    np.testing.assert_allclose(m['rmse_native'],m['rmse']*3)
    np.testing.assert_allclose(out['support']['target_variance_native'][0,:,0],[18.,4.5,2.25])
    assert (out['support']['full_patches'] == 14).all()
    assert not out['support']['nonfinite_prediction_points'].any()


def test_hb_polynomial_derivatives_and_failed_color_are_retained():
    t = np.arange(300)/10
    y = np.stack([t*t/100 + np.sin(t),-.4*t + .5*np.cos(t)],axis=-1)[None]
    b = batch_from_wave(y,2)
    p = dict(sum=b['x'].copy(),S=b['s'].copy(),O=b['o'].copy())
    p['O'][0,4,9] = np.nan
    out = joint_reconstruction_metrics(p,b,2,[2.])
    m = out['metrics']
    np.testing.assert_allclose(m['slope_correlation'][0,:2],1.,atol=1e-12)
    np.testing.assert_allclose(m['curvature_correlation'][0,:2],1.,atol=1e-12)
    np.testing.assert_allclose(m['slope_rmse_native_per_s'][0,:2],0.,atol=1e-12)
    assert np.isnan(m['nmse'][0,2,1]) and m['nmse'][0,2,0] == 0.
    summary = joint_reconstruction_summary(m,out['support'],[(0,0,0)],
        [dict(dataset='d',subject='s',record='r',parent='p')])
    assert summary['support']['nonfinite_prediction_item_color_cells'] == 1
    assert summary['support']['failed_items'][0]['component'] == 'O'
    assert summary['input_distributions']['metrics']['nmse'][5]['nonfinite'] == 1


def test_wrong_units_or_nonfinite_scored_target_fail_before_metrics():
    b = batch_from_wave(np.ones((1,300,2)),2)
    p = dict(sum=b['x'],S=b['s'],O=b['o'])
    with pytest.raises(ValueError,match='positive frozen'):
        joint_reconstruction_metrics(p,b,2,[0.])
    b['s'][0,0,0] = np.nan
    with pytest.raises(ValueError,match='target on declared'):
        joint_reconstruction_metrics(p,b,2,[1.])


def test_rms_gain_keeps_nonzero_tiny_targets_and_marks_exact_zero_undefined():
    y = np.ones((2,300,2))*1e-14
    y[0] = 0.
    b = batch_from_wave(y,2)
    p = dict(sum=b['x'].copy(),S=b['s'].copy(),O=b['o'].copy())
    out = joint_reconstruction_metrics(p,b,2,[1.,1.])
    assert np.isnan(out['metrics']['rms_gain'][0]).all()
    np.testing.assert_allclose(out['metrics']['rms_gain'][1],1.,atol=1e-12)
    assert (out['metrics']['nmse'] == 0).all()


def test_native_units_stay_dataset_specific_when_refed_unit_is_unknown():
    b = batch_from_wave(np.ones((2,6000,1)),1)
    p = dict(sum=b['x'],S=b['s'],O=b['o'])
    out = joint_reconstruction_metrics(p,b,1,[2.,3.])
    rows = [dict(dataset='voltage',subject='a',record='r',parent='p',source_unit='uV'),
            dict(dataset='refed',subject='b',record='r',parent='p',source_unit='unknown_REFED_EEG_export')]
    summary = joint_reconstruction_summary(out['metrics'],out['support'],[(0,0,0),(1,0,0)],rows)
    native = summary['metrics']['rmse_native']
    assert native['units_by_dataset'] == {'refed':['unknown_REFED_EEG_export'],'voltage':['uV']}
    assert native['macro_unit'] is None
    assert not native['cross_unit_macro_interpretable']
    assert summary['metrics']['band_8_13_target_energy_native']['units_by_dataset']['voltage'] == ['uV^2']
