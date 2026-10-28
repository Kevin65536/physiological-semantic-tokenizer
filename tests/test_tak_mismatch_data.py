"""All-data inventory, clock support and feature preparation using small fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import tak_mismatch as data
from src.inference.observation_baselines import eeg_band_power


def geometry(*, refed=False, visual=False):
    frame = 'refed_paper_layout_2d_v1' if refed else 'EEGLAB_CED_cartesian' if visual else 'fixture'
    units = 'normalized_figure_radius' if refed else 'normalized_head_unit'
    common = dict(coordinate_system=frame, coordinate_units=units)
    eeg = [dict(common,channel_name=f'E{i:02}',x=i/20,y=1.,z=0.) for i in range(16)]
    hb = [dict(common,channel_name=f'H{i}_{role}',x=i/4,y=1.,z=0.)
          for i in range(2) for role in ('HbO','HbR')]
    if visual:
        for r in hb:r['coordinate_system']='referenced_visual_eeg_head_coordinates'
    return dict(eeg=eeg,fnirs=hb)


def record_fixture(root, *, ds='eeg_fnirs_single_trial', eeg_seconds=180., hb_seconds=185.):
    subject, record = ('1','video_1') if ds=='refed' else ('subject_01','session_00')
    admitted=root/data.REGISTERED_DATASETS[ds].default_root
    source=admitted/'fixture_EEG.mat'
    g=geometry(refed=ds=='refed')
    manifest=dict(source_files=[dict(path=str(source),sha256='fixture_only')],
        array_shapes=dict(eeg=[round(eeg_seconds*200),16],fnirs=[round(hb_seconds*10),4]),
        measurement=dict(eeg_channel_names=[r['channel_name'] for r in g['eeg']],
                         fnirs_channel_names=[r['channel_name'] for r in g['fnirs']]))
    return SimpleNamespace(dataset_id=ds,canonical_subject_id=subject,subject=subject,
        base_record_id=record,record_id=record+'_hbo_hbr' if ds=='refed' else record,
        join_key=f'{ds}|{subject}|{record}',manifest=manifest),g,source


def events(offset=5., *, supports=None):
    return [dict(event_index=i,eeg_time_ms=t*1000,fnirs_time_ms=(t+offset)*1000,
        metadata={} if supports is None else dict(alignment_support_ms=supports))
        for i,t in enumerate((30.,60.,90.))]


def reports(case='stable_fixed_offset'):
    return [dict(alignment_case=case,label_sequence_match=True)]


@pytest.mark.parametrize('refed,visual',[(False,False),(True,False),(False,True)])
def test_registered_template_frames_preserve_native_pair_order_and_controls(refed,visual):
    g=geometry(refed=refed,visual=visual)
    pairs=data.select_tak_mismatch_channels(g['eeg'],g['fnirs'],seed=17)
    assert [p['id'] for p in pairs]==['H0','H1']
    for p in pairs:
        assert p['status']=='available'
        assert list(p['channel_sets'])==list(data.CHANNEL_CANDIDATES)
        assert all(len(x)==len(set(x))==6 for x in p['channel_sets'].values())
        assert not set(p['channel_sets']['random6']) & set(p['channel_sets']['nearest6'])
        assert not set(p['channel_sets']['next6']) & set(p['channel_sets']['nearest6'])
    reordered=data.select_tak_mismatch_channels(g['eeg'][::-1],g['fnirs'],seed=17)
    assert [p['channel_set_names'] for p in pairs]==[p['channel_set_names'] for p in reordered]


def test_invalid_pair_is_retained_with_reason_not_replaced():
    g=geometry();g['fnirs'][1]['x']=np.nan
    pairs=data.select_tak_mismatch_channels(g['eeg'],g['fnirs'])
    assert len(pairs)==2
    assert pairs[0]['status']=='unavailable' and pairs[0]['channel_sets']=={}
    assert pairs[1]['status']=='available'


def test_Visual_probe_identity_separates_PCA_sites_but_not_subjects():
    g=geometry(visual=True)
    names=[r['channel_name'] for r in g['fnirs']]
    def pair_record(subject,probe,part=1):
        native=SimpleNamespace(dataset_id='visual_cognitive_motivation',
            base_record_id=f'{subject}_Part{part}_Probe{probe}')
        return dict(dataset=native.dataset_id,pairs=data._pair_inventory(g,names,seed=17,record=native))
    one=pair_record('S01',1);same=pair_record('S02',1,2);other=pair_record('S01',2)
    assert [p['id'] for p in one['pairs']]==['Probe1:H0','Probe1:H1']
    assert [p['id'] for p in other['pairs']]==['Probe2:H0','Probe2:H1']
    assert [p['name'] for p in one['pairs']]==['H0','H1']
    assert one['pairs'][0]['channel_set_names']==same['pairs'][0]['channel_set_names']
    assert one['pairs'][0]['channel_sets']['random6']==same['pairs'][0]['channel_sets']['random6']
    assert data.require_fixed_tak_coordinate_groups([one,same,other])==4


@pytest.mark.parametrize('change',['ordered_electrodes','geometry_frame','geometry_version'])
def test_coordinate_group_rejects_changed_electrode_slots_or_geometry(change):
    g=geometry();names=[r['channel_name'] for r in g['fnirs']]
    one=dict(dataset='eeg_fnirs_single_trial',pairs=data._pair_inventory(g,names,seed=17))
    two=deepcopy(one);p=two['pairs'][0]
    if change=='ordered_electrodes':p['channel_set_names']['nearest6'].reverse()
    elif change=='geometry_frame':p['coordinate_system']='different_fixture'
    else:p['geometry_versions']=['different_version']
    with pytest.raises(ValueError,match='PCA coordinate group changes'):
        data.require_fixed_tak_coordinate_groups([one,two])


def test_continuous_tiles_use_segment_offset_and_keep_tail(tmp_path):
    r,_,_=record_fixture(tmp_path,eeg_seconds=184.3,hb_seconds=190.)
    result=data.continuous_tak_support(r,events(5.04),reports())
    assert result['status']=='available' and len(result['windows'])==6
    assert result['tail_seconds']==pytest.approx(10.)
    assert result['windows'][0]['status']=='hb_only'
    paired=[w for w in result['windows'] if w['status']=='available']
    first,last=paired[0],paired[-1]
    assert first['hb_start_s']==30 and first['eeg_start_s']==pytest.approx(24.96)
    assert first['eeg_filter_start_s']>=0
    assert last['eeg_filter_end_s']<=184.3
    assert all(w['actual_context_s']==90 for w in paired)
    assert all(np.isclose((w['eeg_start_s']-w['eeg_filter_start_s'])*10,round((w['eeg_start_s']-w['eeg_filter_start_s'])*10)) for w in paired)
    assert [w['hb_start_s'] for w in result['windows']]==[0,30,60,90,120,150]


def test_piecewise_clock_gap_is_not_joined_or_filtered_across(tmp_path):
    r,_,_=record_fixture(tmp_path,eeg_seconds=250.,hb_seconds=280.)
    one=events(5.,supports=dict(eeg=[None,100000.],fnirs=[None,105000.]))
    two=events(25.,supports=dict(eeg=[140000.,None],fnirs=[165000.,None]))
    result=data.continuous_tak_support(r,one+two,reports('piecewise_constant_offset'))
    assert len(result['segments'])==2
    for w in result['windows']:
        if w['status']!='available':continue
        if w['segment_index']==0:
            assert w['eeg_start_s']+30<=100 and w['eeg_filter_end_s']<=100
        else:
            assert w['eeg_start_s']>=140 and w['eeg_filter_start_s']>=140
            assert w['hb_start_s']-w['eeg_start_s']==25


def test_short_refed_video_keeps_complete_actual_context(tmp_path):
    r,_,_=record_fixture(tmp_path,ds='refed',eeg_seconds=60.,hb_seconds=60.1)
    result=data.continuous_tak_support(r,events(0.),reports('shared_segment_index_no_marker_stream'))
    assert len(result['windows'])==2
    assert all(w['actual_context_s']==60 and w['status']=='available' for w in result['windows'])
    assert all(w['eeg_filter_start_s']==0 and w['eeg_filter_end_s']==60 for w in result['windows'])


@pytest.fixture
def preparation(tmp_path,monkeypatch):
    r,g,source=record_fixture(tmp_path,eeg_seconds=180.,hb_seconds=185.)
    idx=SimpleNamespace(records=[r],records_by_join_key={r.join_key:[r]},
        events_by_join_key={r.join_key:events()},reports_by_join_key={r.join_key:reports()})
    monkeypatch.setattr(data,'CleanPhysiologyCacheIndex',lambda *_:idx)
    class Geometry:
        def __init__(self,*_,**__):pass
        def for_channels(self,*,record,modality,channel_names):return deepcopy(g[modality])
    monkeypatch.setattr(data,'ChannelGeometryIndex',Geometry)
    cfg=dict(seed=17,data=dict(datasets=['eeg_fnirs_single_trial'],folds=5))
    plan=data.inventory_tak_mismatch(cfg,tmp_path)
    row=plan['records'][0]
    voltage=np.random.default_rng(8).normal(size=(36000,16))
    clock=np.arange(1850)/10
    hb=np.stack([np.column_stack((np.sin(clock/8+i),np.cos(clock/7-i))) for i in range(2)],axis=1)
    native=SimpleNamespace(values=voltage,sample_rate_hz=200.,channel_names=row['eeg_channel_names'],
        source_path=source,native_unit='uV',unit_evidence='synthetic',reference='fixture')
    cached=dict(values=hb,time_s=clock,channel_names=row['hb_channel_names'],sample_rate_hz=10.,
        valid_mask=np.ones_like(hb,bool),provenance=dict(schema='unified_cached_fnirs_v1'))
    reads=[]
    def eeg_reader(*_):reads.append('eeg');return native
    def hb_reader(*_):reads.append('hb');return cached
    monkeypatch.setattr(data,'load_native_eeg_record',eeg_reader)
    monkeypatch.setattr(data,'load_cached_fnirs_record',hb_reader)
    return SimpleNamespace(root=tmp_path,record=r,geometry=g,source=source,index=idx,
        row=row,plan=plan,voltage=voltage,hb=hb,clock=clock,native=native,cached=cached,reads=reads)


def test_inventory_is_array_free_and_includes_independent_Hb_denominator(preparation):
    f=preparation
    assert not f.reads and f.plan['measured_arrays_read']==0
    assert len(f.row['windows'])==6
    assert 'hb_only_windows' not in f.row
    assert f.plan['summary']['eeg_fnirs_single_trial']['planned_pair_windows']==12
    assert f.row['hb_only_tail_seconds']==5


def test_prepared_record_shares_channels_and_preserves_native_feature_math(preparation):
    f=preparation
    result=data.prepare_tak_mismatch_record(f.root,f.row,index=f.index,seed=17)
    assert f.reads==['hb','eeg']
    assert result['eeg_logpower'].shape==(6,300,16,5)
    assert result['hb'].shape==(6,300,2,2)
    expected=np.log(eeg_band_power(f.voltage[:18000],bands=data.BANDS_HZ))[250:550]
    expected-=expected[:50].mean(0)
    np.testing.assert_allclose(result['eeg_logpower'][1],expected,rtol=0,atol=1e-14)
    expected_hb=f.hb[:300]-f.hb[:50].mean(0)
    np.testing.assert_allclose(result['hb'][0],expected_hb,atol=2e-15)
    assert result['continuous_hb'] is f.hb
    assert result['hb_bin_phase_s'].shape==(6,)
    assert not result['metadata']['failures']


@pytest.mark.parametrize('field',['source_identity','windows','pairs','subject'])
def test_frozen_identity_changes_fail_before_any_array_access(preparation,field):
    f=preparation;row=deepcopy(f.row)
    if field=='source_identity':row[field][0]['sha256']='drift'
    elif field=='windows':row[field][0]['hb_start_s']+=.1
    elif field=='pairs':row[field][0]['channel_sets']['nearest6'][0]=12
    else:row[field]='other'
    with pytest.raises(ValueError):data.prepare_tak_mismatch_record(f.root,row,index=f.index,seed=17)
    assert not f.reads


def test_Hb_only_survives_EEG_failure_and_support_mask_wins_over_finite_placeholders(preparation,monkeypatch):
    f=preparation
    f.cached['valid_mask'][:,0,1]=False
    def fail(*_):raise ValueError('synthetic failed EEG decode')
    monkeypatch.setattr(data,'load_native_eeg_record',fail)
    result=data.prepare_tak_mismatch_record(f.root,f.row,index=f.index,seed=17)
    assert np.isnan(result['eeg_logpower']).all()
    assert not result['eeg_valid_mask'].any()
    assert np.isnan(result['hb'][:,:,0,1]).all()
    assert np.isfinite(result['hb'][:,:,1,:]).all()
    first=data.prepared_tak_hb_window(result,f.row['windows'][0])
    assert not first['valid_mask'][:,0,1].any() and np.isnan(first['hb'][:,0,1]).all()
    np.testing.assert_allclose(first['hb'][:,1],f.hb[:300,1]-f.hb[:50,1].mean(0))


def test_nonadmitted_EEG_clock_keeps_Hb_population_without_EEG_read(preparation):
    f=preparation
    f.index.reports_by_join_key[f.record.join_key]=reports('continuous_drift')
    plan=data.inventory_tak_mismatch(dict(seed=17,data=dict(datasets=['eeg_fnirs_single_trial'])),f.root)
    row=plan['records'][0]
    assert row['status']=='hb_only' and row['hb_only_admitted']
    assert len(row['windows'])==6 and all(w['status']=='hb_only' for w in row['windows'])
    result=data.prepare_tak_mismatch_record(f.root,row,index=f.index,seed=17)
    assert f.reads==['hb'] and result['eeg_logpower'].shape==(6,300,16,5)
    assert result['continuous_hb'].shape==(1850,2,2)


def test_Hb_gap_and_session_identity_survive_independent_EEG_admission(preparation):
    f=preparation
    anchors=[dict(event_index=i,eeg_time_ms=t*1000,fnirs_time_ms=(t+5)*1000,
        metadata=dict(alignment_support_ms=s)) for i,(t,s) in enumerate([
            (30.,dict(eeg=[None,55000.],fnirs=[None,60000.])),
            (100.,dict(eeg=[85000.,None],fnirs=[90000.,None]))])]
    f.index.events_by_join_key[f.record.join_key]=anchors
    f.index.reports_by_join_key[f.record.join_key]=reports('piecewise_constant_offset')
    plan=data.inventory_tak_mismatch(dict(seed=17,data=dict(datasets=['eeg_fnirs_single_trial'])),f.root)
    row=plan['records'][0]
    assert len(row['windows'])==6 and len(row['continuous_hb_segments'])==2
    assert not row['windows'][2]['hb_support_admitted']
    assert row['windows'][2]['status']=='unavailable'
    assert row['windows'][0]['native_support_id']!=row['windows'][3]['native_support_id']
    result=data.prepare_tak_mismatch_record(f.root,row,index=f.index,seed=17)
    assert not result['continuous_hb_gap_valid_mask'][600:900].any()
    assert result['continuous_hb_observation_valid_mask'][600:900].all()
    assert not result['hb_valid_mask'][2].any() and np.isnan(result['hb'][2]).all()
    assert result['metadata']['continuous_hb_segments']==row['continuous_hb_segments']


def test_touching_Hb_supports_remain_separate_but_EEG_only_change_does_not_split(tmp_path):
    r,_,_=record_fixture(tmp_path)
    anchors=[dict(event_index=i,eeg_time_ms=1000.,fnirs_time_ms=2000.,metadata=dict(
        alignment_support_ms=dict(eeg=es,fnirs=hs))) for i,(es,hs) in enumerate([
            ([None,55000.],[None,60000.]),([55000.,None],[60000.,None])])]
    segments=data.continuous_hb_support_segments(r,anchors)
    assert len(segments)==2 and segments[0]['end_s']==segments[1]['start_s']==60
    anchors[1]['metadata']['alignment_support_ms']['fnirs']=[None,60000.]
    assert len(data.continuous_hb_support_segments(r,anchors))==1


def test_cached_Hb_clock_mismatch_rejected_before_EEG_decode(preparation):
    f=preparation
    f.cached['time_s']=f.clock.copy();f.cached['time_s'][100]+=.001
    with pytest.raises(ValueError,match='regular 10 Hz clock'):
        data.prepare_tak_mismatch_record(f.root,f.row,index=f.index,seed=17)
    assert f.reads==['hb']


def test_missing_EEG_geometry_retains_Hb_without_EEG_decode(preparation):
    f=preparation
    f.geometry['eeg'][0]['x']=None
    plan=data.inventory_tak_mismatch(dict(seed=17,data=dict(datasets=['eeg_fnirs_single_trial'])),f.root)
    row=plan['records'][0]
    assert row['status']=='hb_only' and all(p['status']=='unavailable' for p in row['pairs'])
    result=data.prepare_tak_mismatch_record(f.root,row,index=f.index,seed=17)
    assert f.reads==['hb'] and result['hb_valid_mask'].all()


def test_wrong_matching_never_calls_mixed_or_unknown_conditions_equal(tmp_path):
    r,_,_=record_fixture(tmp_path,eeg_seconds=180.,hb_seconds=180.)
    anchors=[dict(event_index=i,eeg_time_ms=t*1000,fnirs_time_ms=t*1000,label=condition,
        metadata=dict(task='fixture_task',condition_label=condition))
        for i,(t,condition) in enumerate([(0,'A'),(20,'A'),(40,'B'),(80,'unknown')])]
    windows=data.continuous_tak_support(r,anchors,reports())['windows']
    assert windows[0]['wrong_pairing_eligible'] and windows[0]['condition']=='A'
    assert not windows[1]['wrong_pairing_eligible'] and windows[1]['condition'] is None
    assert not windows[2]['wrong_pairing_eligible'] and windows[2]['condition'] is None
