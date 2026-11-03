"""Retained fixed residual geometry and controlled EOG scoring contracts."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.tak_mismatch import fixed_window_physiology_features, analyze_fixed_physiology_features


def fixture():
    t=np.arange(300)*.1
    y=np.column_stack((np.sin(t),np.cos(t/3),-np.cos(t/3)))
    mask=np.ones((300,3),bool);mask[:50]=False;mask[299:]=False
    h=y.copy();h[:,0]=.5*y[:,0];h[:,1:]=0.
    sd=np.ones(3)
    row=dict(hb_only_success=True,eeg_only_success=True,D1=np.sqrt(np.mean(y[50:299,1:]**2)),
        D_EH=np.sqrt(np.mean((y[50:299,0]-h[50:299,0])**2)))
    return dict(time_s=t,target=y,sd=sd,score_mask=mask,hb_only_prediction=h,
        driver_e=y[:,0],driver_h=.5*y[:,0]),row


def test_fixed_residual_moments_preserve_bias_and_direction_energy():
    a,r=fixture();f=fixed_window_physiology_features(a,r)
    assert f['hb_residual_total_ms']==pytest.approx(f['hb_residual_bias_ms']+f['hb_residual_centered_ms'])
    assert f['hb_residual_common_ms']==pytest.approx(0.)
    assert f['hb_residual_common_ms']+f['hb_residual_difference_ms']==pytest.approx(2*f['hb_residual_total_ms'])
    assert f['hb_residual_cov_or']==pytest.approx(-f['hb_residual_var_o'])
    assert f['drive_e_h_corr']==pytest.approx(1.)


def test_failed_view_never_becomes_valid_from_finite_arrays():
    a,r=fixture();r.update(hb_only_success=False,eeg_only_success=False)
    f=fixed_window_physiology_features(a,r)
    assert 'hb_residual_total_ms' not in f and 'drive_e_h_corr' not in f


def test_retained_metric_reproduction_rejects_wrong_arrays():
    a,r=fixture();r['D1']*=2
    with pytest.raises(ValueError,match='reproduce retained fixed D1'):fixed_window_physiology_features(a,r)


def test_eog_lag_uses_same_149_nonwrapping_bins():
    a,r=fixture();rng=np.random.default_rng(11)
    a['target'][:,0]=rng.normal(size=300)
    values=(a['target'][:,0]-a['target'][50:299,0].mean())**2
    ocular=dict(rms_combined=1.,rms_10hz=np.repeat(values[:,None],2,axis=1),valid_mask=np.ones((300,2),bool))
    f=fixed_window_physiology_features(a,r,ocular=ocular)
    assert f['eog_lag_EEG_proxy_amplitude_bins']==149
    assert f['eog_lag_EEG_proxy_amplitude_rho0']==pytest.approx(1.)
    assert f['eog_lag_EEG_proxy_amplitude_contrast']>.8
    ocular['valid_mask'][50]=False
    f=fixed_window_physiology_features(a,r,ocular=ocular)
    assert f['eog_lag_EEG_proxy_amplitude_bins']==148


def test_counterfactual_keeps_fixed_readout_and_scale():
    a,r=fixture();raw=a['target'][:,0].copy();clean=a['hb_only_prediction'][:,0].copy()
    cf=dict(raw_proxy=raw,clean_proxy=clean,amplitude_proxy=.75*raw)
    f=fixed_window_physiology_features(a,r,counterfactual=cf)
    assert f['eog_counterfactual_bins']==249
    assert f['eog_raw_DEH_mse']==pytest.approx(r['D_EH']**2)
    assert f['eog_clean_DEH_mse']==0.
    assert 0<f['eog_amplitude_DEH_mse']<f['eog_raw_DEH_mse']
    cf['raw_proxy']=raw+1e-5
    with pytest.raises(ValueError,match='reproduce fixed target'):fixed_window_physiology_features(a,r,counterfactual=cf)


def test_shape_subject_ratio_uses_averaged_moments_not_average_ratios():
    panel=pd.DataFrame([dict(id=str(i),dataset='fixture',subject='s',task_label='t',record_id=f'r{i}',
        pair='p',native_support_id=f'n{i}',window_start_s=0.,D1=1.,D_EH=np.nan,D_drive_H=np.nan,zero_NMSE=1.) for i in range(2)])
    features=pd.DataFrame([dict(id='0',hb_residual_total_ms=1.,hb_residual_bias_ms=1.,hb_residual_centered_ms=0.,
        hb_residual_common_ms=2.,hb_residual_difference_ms=0.),
        dict(id='1',hb_residual_total_ms=9.,hb_residual_bias_ms=0.,hb_residual_centered_ms=9.,hb_residual_common_ms=0.,hb_residual_difference_ms=18.)])
    out=analyze_fixed_physiology_features(panel,features,config={})
    shapes=out['shape_subjects'].set_index('feature').value
    assert shapes.Hb_bias_fraction==pytest.approx(.1)
    assert shapes.Hb_common_direction_fraction==pytest.approx(.1)
    assert np.isnan(shapes.Hb_residual_OR_correlation)
    assert out['ocular_associations'].empty


def test_complete_panel_denominator_is_required():
    with pytest.raises(ValueError,match='denominator'):
        analyze_fixed_physiology_features(pd.DataFrame({'id':['a','b']}),pd.DataFrame({'id':['a']}),config={})


def test_ocular_effects_count_subjects_and_amplitude_control_can_remove_apparent_gain():
    panel=[];features=[]
    for subject in range(6):
        for window in range(12):
            for pair in range(3):
                identity=f'{subject}_{window}_{pair}';value=window+1.
                panel.append(dict(id=identity,dataset='eeg_fnirs_single_trial',subject=str(subject),
                    task_label='task',record_id=f'r{subject}',native_support_id='n',pair=str(pair),
                    window_start_s=window*30.,D1=value,D_EH=value,D_drive_H=value,zero_NMSE=value**2))
                features.append(dict(id=identity,eog_rms=value,eeg_proxy_ms=value**2,
                    eog_raw_DEH_mse=9.,eog_clean_DEH_mse=4.,eog_amplitude_DEH_mse=4.))
    result=analyze_fixed_physiology_features(pd.DataFrame(panel),pd.DataFrame(features),
        config=dict(bootstrap_samples=20,permutation_samples=100,seed=3))
    table=result['ocular_associations']
    direct=table[table.analysis=='cross_window']
    assert (direct.subjects==6).all() and (direct['mean']==1.).all()
    partial=table[table.analysis=='cross_window_partial']
    assert (partial.subjects==0).all()  # Exact amplitude collinearity is undefined.
    cf=table[table.analysis=='counterfactual'].set_index('contrast')
    assert cf.loc['clean_minus_raw','mean']==-5.
    assert cf.loc['clean_minus_amplitude','mean']==0.
    assert cf.loc['clean_minus_amplitude','p_value']==1.
    records=result['ocular_records']
    assert (records.loc[records.analysis=='cross_window','available_windows']==12).all()
