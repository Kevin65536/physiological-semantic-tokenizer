"""Spatial tau/beta software checks use synthetic data and metadata only."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


@pytest.fixture
def configuration(tmp_path):
    cfg=yaml.safe_load((runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_topography_v1.yaml').read_text())
    parent=tmp_path/cfg['parent'];parent.mkdir(parents=True)
    _,core=runner.tak_repeatability_config(None)
    core['fitting']['reference']['neurovascular_gain']=1.
    core['fitting']['bounds']['neurovascular_gain']=[.1,10.]
    core['arms'].append(dict(name='tak_tau_beta',hemodynamics='tak',initial='volume_tied',tau_v=2,
        fit=['tau','neurovascular_gain'],log_sd=[.125,.125],start_values=[[2.,1.],[1.,.5],[3.5,2.]],warm='tak_fixed'))
    (parent/'resolved_config.json').write_text(json.dumps(dict(schema='shared_driver_tak_task_parameters_v1',data={'protected_data':'forbidden'})))
    (parent/'resolved_model_config.json').write_text(json.dumps(core))
    (parent/'measured_summary.json').write_text(json.dumps({'status':'completed'}))
    return runner.tak_topography_config(None,tmp_path)


def test_spatial_contract_preserves_fit_pair_clock_and_complete_synthetic_denominator(configuration):
    cfg,core=configuration
    assert [a['fit'] for a in core['arms']]==[[],['tau','neurovascular_gain']]
    assert core['arms'][1]['log_sd']==[.125,.125] and core['tensor']['dt_s']==.1
    jobs=runner.tak_topography_specs(core,'synthetic')
    assert len(jobs)==96 and len({j['key'] for j in jobs})==96
    assert cfg['screen']['candidates_per_dataset']==3 and cfg['model']['no_site_normalization']


def test_geometry_projection_aligns_left_anterior_in_both_registered_3d_frames():
    a=runner._tak_topography_xy(dict(x=-.309,y=.95,z=0,coordinate_system='dataset_mnt_pos_3d'))
    b=runner._tak_topography_xy(dict(x=.95,y=.309,z=-.03,coordinate_system='referenced_visual_eeg_head_coordinates'))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(runner._tak_topography_xy(dict(x=-.2,y=.8,z=0,coordinate_system='refed_paper_layout_2d_v1')),[-.2,.8])
    with pytest.raises(ValueError,match='Unsupported'):
        runner._tak_topography_xy(dict(x=1,y=2,z=3,coordinate_system='unknown'))


def test_screen_score_is_unit_invariant_and_penalizes_nonrepeating_spatial_patterns():
    a={str(i):float(i+1) for i in range(8)};b={k:v*2 for k,v in a.items()}
    good=runner._tak_topography_screen_score(a,b)
    scaled=runner._tak_topography_screen_score({k:v*100 for k,v in a.items()},{k:v*.001 for k,v in b.items()})
    assert good['eligible'] and good['score']>0
    assert scaled['score']==pytest.approx(good['score'])
    reverse={str(i):float(8-i) for i in range(8)}
    assert runner._tak_topography_screen_score(a,reverse)['score']==0
    unavailable=dict(b);unavailable.update({'0':np.nan,'1':np.nan})
    r=runner._tak_topography_screen_score(a,unavailable)
    assert not r['eligible'] and r['planned_sites']==8 and r['common_sites']==6


def test_spatial_truth_constant_across_repeats_but_driver_changes(configuration):
    _,core=configuration;values={}
    for scenario in ['uniform','structured']:
        for node in [0,1]:
            pair=[]
            for repeat in [0,1]:
                case=dict(subject=0,repeat=repeat,task='A' if repeat==0 else 'B',scenario=scenario,spatial_node=node)
                y,op,sd,truth=runner.tak_task_parameter_synthetic(core,case)
                assert y.shape==(900,3) and op.shape==(2700,2700)
                pair.append(truth)
            assert (pair[0]['tau'],pair[0]['beta'])==(pair[1]['tau'],pair[1]['beta'])
            assert not np.array_equal(pair[0]['driver'],pair[1]['driver'])
            values[(scenario,node)]=(pair[0]['tau'],pair[0]['beta'])
    assert values[('uniform',0)]==values[('uniform',1)]
    assert values[('structured',0)]!=values[('structured',1)]


def test_screen_stops_before_measured_metadata_or_arrays_if_synthetic_incomplete(tmp_path,monkeypatch):
    (tmp_path/'synthetic_summary.json').write_text(json.dumps(dict(status='incomplete',terminal_rows=95)))
    monkeypatch.setattr(runner,'tak_topography_plan',lambda *args:pytest.fail('No measured admission before software/synthetic completion'))
    with pytest.raises(ValueError,match='synthetic matrix'):
        runner.tak_topography_screen({},tmp_path,tmp_path,{},1)


def test_modified_protected_scope_is_rejected_before_parent_reads(tmp_path):
    cfg=yaml.safe_load((runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_topography_v1.yaml').read_text())
    cfg['data']['protected_data']='allowed';path=tmp_path/'modified.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Unchanged versioned'):
        runner.tak_topography_config(path,tmp_path)


def test_screen_honors_both_species_masks_without_losing_other_channels(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import src.data.clean_physiology_cache as cache
    import src.data.unified_physiology as reader
    time=np.arange(1800)*.1;values=np.sin(time[:,None,None]/10)*np.array([1.,2.,3.])[None,:,None]*np.array([1.,-.3])[None,None,:]
    mask=np.ones(values.shape,bool);mask[:,2,1]=False
    record=SimpleNamespace(join_key='fixture')
    monkeypatch.setattr(cache,'CleanPhysiologyCacheIndex',lambda *a:SimpleNamespace(records=[record]))
    monkeypatch.setattr(reader,'load_cached_fnirs_record',lambda *a:dict(time_s=time,values=values,valid_mask=mask,provenance={'fixture':True}))
    pairs=[dict(native_index=i,status='available',xy=[i/4,0],site_id=str(i)) for i in range(3)]
    subject=dict(id='fixture_s',dataset='fixture',subject='s',role='candidate',windows=[dict(id=f'w{i}',join_key='fixture',hb_start_s=i*90,map_repeat=i,pairs=pairs) for i in range(2)])
    cfg=dict(data={'cache_root':'unused'},screen={'minimum_common_site_fraction':.8})
    result=runner._tak_topography_screen_worker((cfg,str(tmp_path),str(tmp_path),subject))
    assert result['common_sites']==2 and result['planned_sites']==3 and not result['eligible']
    assert not result['failures']
    with np.load(tmp_path/'spatial_features/w0.npz') as a:np.testing.assert_array_equal(a['valid'],[True,True,False])


def _map_fixture():
    import pandas as pd
    rows=[]
    for s in range(3):
        for repeat in range(2):
            for node in range(5):
                shape=np.roll(np.arange(1,6),s)[node]
                rows.append(dict(dataset='fixture',subject=str(s),map_repeat=repeat,location_id=str(node),
                    complete=True,tau=float(shape*2**repeat),beta=float(shape*3**repeat),x=node/3-1,y=node/3-1,
                    hb_relative_rmse=.1,member_sites=json.dumps([str(node)]),member_count=1))
    return pd.DataFrame(rows)


def test_map_shape_comparison_separates_global_level_from_topography():
    result=runner._tak_topography_map_statistics(_map_fixture());r=result['repeatability']
    np.testing.assert_allclose(r.spearman,1)
    np.testing.assert_allclose(r.centered_log_rms,0,atol=1e-15)
    np.testing.assert_allclose(r[r.parameter.eq('tau')].absolute_log_rms,np.log(2))
    np.testing.assert_allclose(r[r.parameter.eq('beta')].absolute_log_rms,np.log(3))
    assert result['between_subject'].centered_log_rms.gt(.1).all()
    assert result['dataset_summary'].selected_subjects.eq(3).all()
    assert result['dataset_summary'].between_within_shape_ratio.isna().all()


def test_missing_locations_and_all_failed_maps_keep_planned_denominators():
    f=_map_fixture();f.loc[(f.subject=='0')&(f.map_repeat==1)&(f.location_id=='0'),'complete']=False
    r=runner._tak_topography_map_statistics(f)['repeatability']
    assert r[r.subject.eq('0')].planned_locations.eq(5).all()
    assert r[r.subject.eq('0')].paired_locations.eq(4).all()
    f['complete']=False;r=runner._tak_topography_map_statistics(f)
    assert r['mean_maps'].empty and r['repeatability'].paired_locations.eq(0).all()
    assert r['dataset_summary'].median_within_spearman.isna().all()


def test_documented_channel_mismatch_traces_local_inputs_and_marks_shared_coordinate():
    audit=dict(schema='visual_channel_identity_audit_v2',channel_comparison=[
        dict(current='Fp1',documented='FP1',raw_channel='A1',matches=True),
        dict(current='P3',documented='CP3',raw_channel='A11',matches=False),
        dict(current=None,documented='FCz',raw_channel='A64',matches=False)])
    refs=[dict(dataset='visual_cognitive_motivation',id=str(i),subject='S01',site_id=str(i),map_repeat=0,eeg_channels=c)
        for i,c in enumerate([['Fp1','P3'],['FP1']])]
    result=runner._tak_topography_identity_impact(dict(records=refs),audit)
    assert result['planned_cells']==2 and result['cells_with_directly_mismatched_local_channels']==1
    assert result['interpretation']=='diagnostic_only_invalid_EEG_identity'
    assert result['cells'][0]['mismatched_local_channels'][0]['actual']=='CP3'
    assert not result['cells'][1]['mismatched_local_channels']
