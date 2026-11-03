"""Native auxiliary EOG alignment and missingness without measured fixtures."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import tak_mismatch_ocular as ocular
from src.data.tak_mismatch import continuous_hb_support_segments, continuous_tak_support
from src.data.registry import REGISTERED_DATASETS


def test_native_rms_uses_score_mean_and_native_derivative_without_bin_normalization():
    rate = 200.
    times = np.arange(6000) / rate
    raw = np.column_stack((17 + 2 * np.sin(2 * np.pi * times),
                           -31 + 3 * np.cos(2 * np.pi * times / 3)))
    result = ocular.native_ocular_window(raw, rate)
    score = raw[1000:5980]
    center = score.mean(axis=0)
    np.testing.assert_allclose(result['rms'], np.std(score, axis=0))
    np.testing.assert_allclose(result['peak_abs'], np.abs(score-center).max(axis=0))
    np.testing.assert_allclose(result['derivative_rms'], np.sqrt(np.mean((np.diff(score, axis=0)*rate)**2, axis=0)))
    np.testing.assert_allclose(result['rms_10hz'],
        np.sqrt(np.mean((raw-center).reshape(300,20,2)**2, axis=1)))
    np.testing.assert_allclose(result['valid_seconds'], [24.9,24.9])
    np.testing.assert_allclose(result['rms_combined'], np.sqrt(np.mean(result['rms']**2)))
    assert result['valid_mask'].all()
    shifted = ocular.native_ocular_window(raw + [1000,-800], rate)
    np.testing.assert_allclose(shifted['rms'], result['rms'], atol=1e-12)


def test_missing_score_sample_retains_other_channel_and_actual_duration():
    raw = np.ones((6000,2))
    raw[2000,0] = np.nan
    result = ocular.native_ocular_window(raw, 200.)
    assert np.isnan(result['rms'][0]) and result['rms'][1] == 0
    assert result['valid_seconds'][0] == pytest.approx(24.895)
    assert not result['valid_mask'][:,0].any() and result['valid_mask'][:,1].all()
    assert np.isnan(result['rms_combined'])
    raw[2000,0] = 1
    raw[5,0] = np.nan
    result = ocular.native_ocular_window(raw, 200.)
    assert result['complete'].all() and result['rms'][0] == 0
    assert not result['valid_mask'][0,0] and result['valid_mask'][1:,0].all()


@pytest.fixture
def native_record(tmp_path, monkeypatch):
    dataset = 'eeg_fnirs_single_trial'
    source = tmp_path / REGISTERED_DATASETS[dataset].default_root / 'fixture.mat'
    state = dict(auxiliary_channel_names=['VEOG','HEOG'], native_sample_rate_hz=200.,
                 signal_branch='single_trial_eeg_artifact_clean_v4')
    manifest = dict(source_files=[dict(path=str(source),sha256='fixture')],
        array_shapes=dict(eeg=[36000,2],fnirs=[1850,2]),
        measurement=dict(eeg_channel_names=['F3','F4'],eeg_preprocessing_state=state))
    record = SimpleNamespace(dataset_id=dataset,canonical_subject_id='subject_01',
        base_record_id='session_00',record_id='session_00',
        join_key=dataset+'|subject_01|session_00',manifest=manifest)
    events = [dict(event_index=i,eeg_time_ms=t*1000,fnirs_time_ms=(t+5.04)*1000,
                   metadata={}) for i,t in enumerate([30.,60.,90.])]
    reports = [dict(alignment_case='stable_fixed_offset',label_sequence_match=True)]
    index = SimpleNamespace(records_by_join_key={record.join_key:[record]},
        events_by_join_key={record.join_key:events},reports_by_join_key={record.join_key:reports})
    coverage = continuous_tak_support(record,events,reports)
    windows = coverage.pop('windows')
    plan = dict(join_key=record.join_key,id=record.join_key,cache_record_id=record.record_id,
        dataset=dataset,subject=record.canonical_subject_id,record=record.base_record_id,
        source_identity=deepcopy(manifest['source_files']),array_shapes=deepcopy(manifest['array_shapes']),
        eeg_channel_names=['F3','F4'],windows=windows,coverage=coverage,
        continuous_hb_segments=continuous_hb_support_segments(record,events))
    times = np.arange(36000)/200.
    native = SimpleNamespace(values=np.zeros((36000,2)),sample_rate_hz=200.,
        channel_names=('F3','F4'),auxiliary_values=np.column_stack((3*np.sin(times),2*np.cos(times))),
        auxiliary_channel_names=('VEOG','HEOG'),source_path=source,native_unit='uV',
        unit_evidence='synthetic header',reference='synthetic reference')
    reads = []
    def loader(*args):
        reads.append(args)
        return native
    monkeypatch.setattr(ocular,'load_native_eeg_record',loader)
    return tmp_path,record,index,plan,native,reads


def test_record_preserves_parent_grid_auxiliary_order_and_hb_only_denominator(native_record):
    root,record,index,plan,native,reads = native_record
    result = ocular.prepare_tak_ocular_record(root,plan,index=index,record=record)
    assert len(reads) == 1
    assert result['metadata']['channels'] == ['HEOG','VEOG']
    assert result['metadata']['native_unit'] == 'uV'
    np.testing.assert_array_equal(result['window_ids'], [w['id'] for w in plan['windows']])
    for i,w in enumerate(plan['windows']):
        if w['status'] != 'available':
            assert np.isnan(result['rms'][i]).all()
            continue
        start = round(w['eeg_filter_start_s']*200) + round((w['eeg_start_s']-w['eeg_filter_start_s'])*10)*20
        np.testing.assert_array_equal(result['native_target_sample_bounds'][i],[start,start+6000])
        np.testing.assert_array_equal(result['native_score_sample_bounds'][i],[start+1000,start+5980])
        expected = native.auxiliary_values[start+1000:start+5980, [1,0]]
        np.testing.assert_allclose(result['rms'][i], np.std(expected,axis=0))
        assert abs(result['target_start_rounding_s'][i]) <= .0025


@pytest.mark.parametrize('change',['source','clock','support'])
def test_changed_parent_identity_fails_before_read(native_record,change):
    root,record,index,plan,native,reads = native_record
    plan = deepcopy(plan)
    if change == 'source': plan['source_identity'][0]['path'] += '.changed'
    elif change == 'clock': plan['windows'][1]['eeg_start_s'] += .1
    else: plan['continuous_hb_segments'][0]['end_s'] -= .1
    with pytest.raises(ValueError,match='changed before'):
        ocular.prepare_tak_ocular_record(root,plan,index=index)
    assert reads == []


def test_unregistered_eog_does_not_open_native_eeg_or_invent_scalp_reference(native_record):
    root,record,index,plan,native,reads = native_record
    record.manifest['measurement']['eeg_preprocessing_state']['auxiliary_channel_names'] = []
    result = ocular.prepare_tak_ocular_record(root,plan,index=index)
    assert reads == []
    assert result['metadata']['reason'] == 'no_registered_native_EOG_auxiliary'
    assert np.isnan(result['rms']).all() and not result['complete'].any()


def test_nonfinite_auxiliary_samples_remain_missing_and_keep_other_channel(native_record):
    root,record,index,plan,native,reads = native_record
    i,w = next((i,w) for i,w in enumerate(plan['windows']) if w['status']=='available')
    start = round(w['eeg_filter_start_s']*200) + round((w['eeg_start_s']-w['eeg_filter_start_s'])*10)*20
    native.auxiliary_values[start+1800,1] = np.nan
    result = ocular.prepare_tak_ocular_record(root,plan,index=index)
    assert not result['complete'][i,0] and result['complete'][i,1]
    assert np.isnan(result['rms'][i,0]) and np.isfinite(result['rms'][i,1])
    assert result['metadata']['windows'][i]['reason'] == 'nonfinite_native_EOG_score_support'


def test_amplitude_control_matches_only_clean_score_moments_without_changing_raw_shape():
    rng=np.random.default_rng(912)
    raw=rng.normal(size=300)
    clean=7+2*rng.normal(size=300)
    matched,state=ocular.amplitude_matched_proxy(raw,clean)
    assert state['score_count']==249 and state['ddof']==0
    assert matched[50:299].mean()==pytest.approx(clean[50:299].mean())
    assert matched[50:299].std()==pytest.approx(clean[50:299].std())
    assert np.corrcoef(raw,matched)[0,1]==pytest.approx(1.)
    matched,state=ocular.amplitude_matched_proxy(raw,np.ones(300)*4)
    np.testing.assert_array_equal(matched,np.full(300,4.))
    assert state['clean_constant']
    matched,state=ocular.amplitude_matched_proxy(np.ones(300),clean)
    assert np.isnan(matched).all() and state['status']=='unavailable_raw_zero_scale'


@pytest.fixture
def counterfactual_record(native_record):
    root,record,index,plan,native,reads=native_record
    rng=np.random.default_rng(67)
    rate=native.sample_rate_hz
    times=np.arange(len(native.values))/rate
    eog=np.column_stack((10*np.sin(2*np.pi*2*times),15*np.sin(2*np.pi*1.3*times+.2)))
    native.auxiliary_values=eog
    native.values=rng.normal(size=(len(times),12))+.6*eog@rng.normal(size=(2,12))
    native.channel_names=tuple(f'E{i:02}' for i in range(12))
    plan['eeg_channel_names']=list(native.channel_names)
    record.manifest['measurement']['eeg_channel_names']=list(native.channel_names)
    record.manifest['array_shapes']['eeg'][1]=12
    plan['array_shapes']=deepcopy(record.manifest['array_shapes'])
    plan['pairs']=[dict(id=f'H{k}',channel_sets=dict(nearest6=list(range(k*6,k*6+6)))) for k in range(2)]
    coords={}
    for pair in plan['pairs']:
        pc=rng.normal(size=30);pc/=np.linalg.norm(pc)
        coords[pair['id']]=dict(pc=dict(nearest6=pc.tolist()),eeg_factor=.17)
    features=np.full((len(plan['windows']),300,12,5),np.nan)
    for i,w in enumerate(plan['windows']):
        if w['status']!='available':continue
        a,b=[round(w[k]*rate) for k in ['eeg_filter_start_s','eeg_filter_end_s']]
        power=ocular.eeg_band_power(native.values[a:b],bands=ocular.BANDS_HZ,sample_rate=rate,target_rate=10.)
        start=round((w['eeg_start_s']-w['eeg_filter_start_s'])*10)
        target=np.log(power[start:start+300]);features[i]=target-target[:50].mean(axis=0)
    prepared=dict(eeg_logpower=features,eeg_proxy_input=np.array(ocular.EEG_PROXY_INPUT_SCHEMA),
        window_ids=np.asarray([w['id'] for w in plan['windows']]),time_s=np.arange(300)/10.)
    return root,record,index,plan,native,reads,coords,prepared


def test_counterfactual_reuses_native_regression_and_frozen_coordinates_without_mutating_raw(counterfactual_record):
    root,record,index,plan,native,reads,coords,prepared=counterfactual_record
    before=native.values.copy(); frozen=deepcopy(coords)
    result=ocular.prepare_tak_ocular_counterfactual(root,plan,coords,['H0','H1'],
        prepared=prepared,index=index)
    assert len(reads)==1 and result['metadata']['status']=='available'
    assert result['metadata']['measured_signal_reads']==1
    np.testing.assert_array_equal(native.values,before)
    assert coords==frozen
    np.testing.assert_array_equal(result['pair_ids'],['H0','H1'])
    assert all(x['status']=='passed' for x in result['metadata']['raw_reproduction'])
    assert max(x['maximum_feature_difference'] for x in result['metadata']['raw_reproduction'])==0
    reg=result['metadata']['regression']
    assert np.all(np.array(reg['removed_variance_fraction'])<=.5)
    assert np.any(np.array(reg['clean_variance'])<np.array(reg['raw_variance']))
    assert reg['fit_sample_count']<=30000
    for i,w in enumerate(plan['windows']):
        if w['status']!='available':
            assert not result['complete'][i].any();continue
        for j in range(2):
            channels=plan['pairs'][j]['channel_sets']['nearest6']
            expected=prepared['eeg_logpower'][i][:,channels,:].reshape(300,30)@np.array(coords[f'H{j}']['pc']['nearest6'])*.17
            np.testing.assert_array_equal(result['raw_proxy'][i,j],expected)
            assert result['complete'][i,j] and result['amplitude_available'][i,j]
            clean=result['clean_proxy'][i,j];amp=result['amplitude_proxy'][i,j]
            assert not np.allclose(clean,expected)
            assert clean[:50].mean()==pytest.approx(0,abs=1e-12)
            assert amp[50:299].mean()==pytest.approx(clean[50:299].mean())
            assert amp[50:299].std()==pytest.approx(clean[50:299].std())


def test_counterfactual_reproduction_failure_prevents_cleanup(counterfactual_record,monkeypatch):
    root,record,index,plan,native,reads,coords,prepared=counterfactual_record
    i=next(i for i,w in enumerate(plan['windows']) if w['status']=='available')
    prepared['eeg_logpower'][i,100,0,0]+=.01
    monkeypatch.setattr(ocular,'detect_ocular_mask',lambda *_:pytest.fail('regression must not start after changed parent feature'))
    with pytest.raises(ValueError,match='does not reproduce'):
        ocular.prepare_tak_ocular_counterfactual(root,plan,coords,['H0'],prepared=prepared,index=index)


def test_counterfactual_incomplete_EOG_does_not_impute_or_report_success(counterfactual_record):
    root,record,index,plan,native,reads,coords,prepared=counterfactual_record
    native.auxiliary_values[0,0]=np.nan
    result=ocular.prepare_tak_ocular_counterfactual(root,plan,coords,['H0'],prepared=prepared,index=index)
    assert result['metadata']['reason']=='incomplete_native_record_EOG_support_no_regression_imputation'
    assert not result['complete'].any() and np.isnan(result['clean_proxy']).all()


def test_counterfactual_missing_registered_EOG_skips_both_prepared_and_native(counterfactual_record):
    root,record,index,plan,native,reads,coords,prepared=counterfactual_record
    record.manifest['measurement']['eeg_preprocessing_state']['auxiliary_channel_names']=[]
    result=ocular.prepare_tak_ocular_counterfactual(root,plan,coords,['H0'],prepared={},index=index)
    assert reads==[] and result['metadata']['reason']=='no_registered_native_EOG_auxiliary'


def test_counterfactual_keeps_original_preparation_failure_without_reprocessing_its_context(counterfactual_record,monkeypatch):
    root,record,index,plan,native,reads,coords,prepared=counterfactual_record
    i=next(i for i,w in enumerate(plan['windows']) if w['status']=='available')
    prepared['eeg_logpower'][i]=np.nan
    original=ocular._context_logpower
    def valid_contexts_only(voltage,window,rate):
        assert window['id']!=plan['windows'][i]['id']
        return original(voltage,window,rate)
    monkeypatch.setattr(ocular,'_context_logpower',valid_contexts_only)
    result=ocular.prepare_tak_ocular_counterfactual(root,plan,coords,['H0'],prepared=prepared,index=index)
    assert not result['complete'][i,0]
    row=next(r for r in result['metadata']['windows'] if r['window_index']==i)
    assert row['reason']=='incomplete_retained_raw_proxy'
