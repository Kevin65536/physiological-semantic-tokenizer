"""Small non-authorized fixtures for Visual exclusion and cache migration."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.event_alignment import read_visual_fnirs_exclusions
from src.data import downstream_input_cache as cache
from src.data.clean_physiology_cache import require_current_cache_manifest


def header(path, flags):
    path.write_text('Header\nSampling Period[s],0.1\nException Ch,'+','.join(map(str,flags))+
        '\nData\n'+','.join([f'CH{i}' for i in range(1,25)]+['Time','Mark'])+'\nTHIS IS NOT A SIGNAL ROW\n')


def fixture(root, excluded_multiplier=1.):
    rng=np.random.default_rng(2);t=np.arange(900)/10
    values=rng.normal(size=(900,24,2))+np.sin(t[:,None,None])
    values[:,:10]*=excluded_multiplier
    source=root/'old';source.mkdir(parents=True)
    files=[root/'test_Oxy.csv',root/'test_Deoxy.csv'];flags=[1]*10+[0]*14
    for p in files:header(p,flags)
    for name,a in dict(fnirs=values.reshape(900,48),fnirs_supported_channels=np.ones(48,bool),
            eeg=np.ones((18000,31)),eeg_supported_channels=np.ones(31,bool),eeg_bad_channel_mask=np.zeros(31,bool)).items():np.save(source/(name+'.npy'),a)
    names=[f'CH{i}_{h}' for i in range(1,25) for h in ['HbO','HbR']]
    key='visual_cognitive_motivation|S01|S01_Probe1'
    m=dict(schema='clean_eeg_fnirs_cache_v2',dataset_id=cache.VISUAL,subject='S01',canonical_subject_id='S01',
        record_id='S01_Probe1',base_record_id='S01_Probe1',join_key=key,record_npz=str(source),
        arrays={k:k+'.npy' for k in ['eeg','fnirs','eeg_supported_channels','fnirs_supported_channels','eeg_bad_channel_mask']},
        array_shapes=dict(eeg=[18000,31],fnirs=[900,48],fnirs_supported_channels=[48]),
        source_files=[dict(path=str(p)) for p in files],measurement=dict(fnirs_channel_names=names,fnirs_component_roles=[h for i in range(24) for h in ['HbO','HbR']],
            fnirs_preprocessing_state=dict(canonical_unit='relative:visual_cognitive_motivation')))
    x=values-values[:50].mean(0);old=.025/np.sqrt(x.reshape(-1,2).var(0).mean())
    calibration=dict(windows=[dict(join_key=key,window_key='w00000',start=0,steps=900,original_pairs=24)],old_factor=old,old_count=900*24)
    cfg=dict(output_measurement='measurement_v6',output_cache='downstream_v2',normalization=dict(schema='visual_subject_common_hb_v1',target_common_sd=.025))
    selection=cache._visual_selection(m)
    return key,m,cfg,selection,calibration,values


def test_header_fails_closed_and_stops_before_signals(tmp_path):
    p=tmp_path/'x.csv';header(p,[0]*23+[1])
    assert read_visual_fnirs_exclusions(p)['excluded_channels']==['CH24']
    p.write_text(p.read_text().replace('Exception Ch','Not Exception Ch'))
    with pytest.raises(ValueError,match='Missing or invalid'):read_visual_fnirs_exclusions(p)


def test_disagreement_rejected_before_array_read(tmp_path):
    key,m,cfg,selection,calibration,values=fixture(tmp_path)
    header(tmp_path/'test_Deoxy.csv',[0]*24)
    Path(m['record_npz'],'fnirs.npy').unlink()
    with pytest.raises(ValueError,match='disagree'):
        cache._visual_migration_subject((str(tmp_path),cfg,'S01',{key:selection},calibration,{key:m},{key:m}))


def test_compaction_scaling_and_excluded_amplitude_isolation(tmp_path):
    factors=[]
    for multiplier in [1.,1000.]:
        root=tmp_path/str(multiplier)
        key,m,cfg,selection,calibration,values=fixture(root,multiplier)
        r=cache._visual_migration_subject((str(root),cfg,'S01',{key:selection},calibration,{key:m},{key:m}))
        low=json.loads(Path(r['measurement_records'][0]).read_text());high=json.loads(Path(r['downstream_records'][0]).read_text())
        require_current_cache_manifest(low)
        assert low['measurement']['fnirs_channel_names'][0]=='CH11_HbO'
        assert low['array_shapes']['fnirs']==[900,28]
        actual=np.load(Path(low['record_npz'])/'fnirs.npy')
        np.testing.assert_array_equal(actual,values[:,10:].reshape(900,28))
        np.testing.assert_allclose(np.load(Path(high['record_npz'])/'fnirs.npy'),actual*r['factor'])
        assert high['measurement']['fnirs_preprocessing_state']['canonical_unit']=='subject_normalized_Hb'
        assert np.load(Path(m['record_npz'])/'fnirs.npy').shape==(900,48)
        factors.append(r['factor'])
        wrong=deepcopy(low);wrong['measurement']['fnirs_channel_names'][0]='CH1_HbO'
        with pytest.raises(ValueError,match='compact'):require_current_cache_manifest(wrong)
    assert factors[0]==factors[1]
