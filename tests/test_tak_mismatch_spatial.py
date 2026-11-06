"""Spatial mismatch diagnostics use small synthetic tables, never retained data."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import tak_mismatch as analysis
from src.data.channel_geometry import project_head_topview


def _fixture(subjects=("clear",), dataset="eeg_fnirs_single_trial", times=6):
    rows, geometry = [], []
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for subject in subjects:
        for record in ("record1", "record2"):
            identity = dataset+"|"+subject+"|"+record
            for i, angle in enumerate(angles):
                pair = f"CH{i+1}"
                value = 1.2+.8*np.cos(angle) if subject == "clear" else 1.2+.8*(-1)**i
                geometry.append(dict(dataset=dataset,subject=subject,record_id=identity,pair=pair,
                    x=.7*np.cos(angle),y=.7*np.sin(angle),location_id=str(i),geometry_status="available"))
                for time in range(times):
                    rows.append(dict(dataset=dataset,subject=subject,record_id=identity,record_unit=identity,
                        task_label="task",pair=pair,window_start_s=30.*time,native_support_id=identity+"|support0",
                        parameter_mode="fixed",D1=value*(1+time*.01),zero_NMSE=2.+.2*i))
    frame = pd.DataFrame(rows)
    native = analysis.aggregate_fixed_metrics(frame,metrics=("D1",))["native_pairs"]
    return frame, native, pd.DataFrame(geometry)


def test_half_maps_drop_odd_middle_and_keep_matched_ratio():
    windows, _, _ = _fixture(times=5)
    for row in range(len(windows)):
        time = int(windows.loc[row,"window_start_s"]//30)
        windows.loc[row,"D1"] = [1,3,99,5,7][time]
        windows.loc[row,"zero_NMSE"] = [2,8,1,18,32][time]
    halves = analysis.spatial_half_metrics(windows)
    a = halves[(halves.period=="early") & (halves.metric=="D1")]
    b = halves[(halves.period=="late") & (halves.metric==analysis.HB_RATIO)]
    np.testing.assert_allclose(a.value,np.sqrt(5))
    np.testing.assert_allclose(b.value,74/50)
    assert a.planned_windows.eq(2).all()
    windows.loc[windows.window_start_s.eq(0),"D1"] = np.nan
    assert analysis.spatial_half_metrics(windows).query("period == 'early'").value.isna().all()


def test_spatial_maps_pool_squared_errors_and_ratio_parts_at_coincident_locations():
    windows, native, geometry = _fixture()
    native = native[native.pair.isin(["CH1","CH2"])].copy()
    geometry = geometry[geometry.pair.isin(["CH1","CH2"])].copy()
    geometry["location_id"] = "same_projection"
    geometry[["x","y"]] = [0.,.7]
    for index,row in native.iterrows():
        numerator = 1. if row.pair=="CH1" else 9.
        denominator = (2. if row.pair=="CH1" else 8.) if row.metric==analysis.HB_RATIO else 1.
        native.loc[index,["aggregation_numerator","aggregation_denominator","value"]] = [numerator,denominator,
            np.sqrt(numerator) if row.metric=="D1" else numerator/denominator]
    _,maps = analysis.aggregate_spatial_maps(native, pd.DataFrame(), geometry)
    full = maps[maps.period.eq("full")]
    np.testing.assert_allclose(full[full.metric.eq("D1")].value,np.sqrt(5))
    np.testing.assert_allclose(full[full.metric.eq(analysis.HB_RATIO)].value,1.)
    assert full.planned_members.eq(2).all()
    assert full.pair_members.eq("CH1;CH2").all()


def test_visual_EEG_suppression_does_not_gate_Hb_maps():
    windows,native,geometry = _fixture(dataset="visual_cognitive_motivation")
    eeg = native[native.metric.eq("D1")].copy()
    eeg["metric"] = "D_EH"
    _,maps = analysis.aggregate_spatial_maps(pd.concat([native,eeg]),analysis.spatial_half_metrics(windows),geometry)
    assert maps[maps.metric.eq("D_EH")].display_value.isna().all()
    assert maps[maps.metric.eq("D1")].display_value.notna().all()
    assert maps.query("period == 'full' and metric == 'D_EH'").value.notna().all()


def test_corrected_visual_schema_makes_EEG_maps_interpretable_without_unlocking_old_evidence():
    from src.data.unified_physiology import VISUAL_CHANNEL_IDENTITY_SCHEMA
    windows, native, geometry = _fixture(dataset='visual_cognitive_motivation')
    eeg = native[native.metric.eq('D1')].copy(); eeg['metric'] = 'D_EH'
    geometry['native_eeg_identity'] = VISUAL_CHANNEL_IDENTITY_SCHEMA
    _, maps = analysis.aggregate_spatial_maps(pd.concat([native, eeg]),
        analysis.spatial_half_metrics(windows), geometry)
    assert maps[maps.metric.eq('D_EH')].display_value.notna().all()
    # A coincident old/new mixture must not acquire the repaired identity.
    geometry.loc[geometry.pair.eq('CH2'), 'native_eeg_identity'] = 'retained_original'
    geometry.loc[geometry.pair.isin(['CH1', 'CH2']), 'location_id'] = 'coincident'
    _, mixed = analysis.aggregate_spatial_maps(pd.concat([native, eeg]),
        analysis.spatial_half_metrics(windows), geometry)
    assert mixed.query("metric == 'D_EH' and location_id == 'coincident'").display_value.isna().all()


def test_partial_time_support_is_visible_but_not_used_as_high_coverage_selection():
    windows,native,geometry = _fixture()
    take = native.record_id.str.endswith("record2")
    native.loc[take,"available_windows"] = 2
    _,maps = analysis.aggregate_spatial_maps(native,analysis.spatial_half_metrics(windows),geometry)
    partial = maps[maps.period.eq("full") & maps.record_unit.str.endswith("record2")]
    assert partial.display_value.notna().all()
    assert not partial.selection_valid.any()
    mean = maps[maps.period.eq("subject_mean")]
    assert mean.display_value.notna().all() and mean.available_records.eq(2).all()


def test_selection_rewards_repeating_neighbour_pattern_and_retains_all_candidates():
    windows,native,geometry = _fixture(subjects=("clear","checkerboard"))
    _,maps = analysis.aggregate_spatial_maps(native,analysis.spatial_half_metrics(windows),geometry)
    records,ranking = analysis.rank_spatial_patterns(maps,examples_per_dataset=1)
    assert len(records)==4 and len(ranking)==2
    assert ranking[ranking.selected].subject.tolist()==["clear"]
    assert ranking.set_index("subject").loc["clear","score"]>0
    assert ranking.set_index("subject").loc["checkerboard","score"]==pytest.approx(0.,abs=1e-12)
    similarity = analysis.spatial_record_similarities(maps,ranking)
    assert similarity.common_locations.eq(12).all()
    np.testing.assert_allclose(similarity.spearman_rho,1.)


def test_missing_half_support_cannot_select_a_subject_or_fill_zero():
    windows,native,geometry = _fixture(times=3)
    halves = analysis.spatial_half_metrics(windows)
    assert halves.empty
    _,maps = analysis.aggregate_spatial_maps(native,halves,geometry)
    records,ranking = analysis.rank_spatial_patterns(maps)
    assert not records.eligible.any() and not ranking.selected.any()
    assert maps[maps.period.eq("full")].display_value.notna().all()


def test_projection_requires_registered_frame_and_preserves_left_anterior():
    a=project_head_topview(dict(x=-.3,y=.9,z=0,coordinate_system="dataset_mnt_pos_3d"))
    b=project_head_topview(dict(x=.9,y=.3,z=0,coordinate_system="referenced_visual_eeg_head_coordinates"))
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match="Unsupported"):
        project_head_topview(dict(x=-.3,y=.9,z=0,coordinate_system="unknown"))
    with pytest.raises(ValueError,match="coincident"):
        analysis.spatial_neighbour_weights(np.array([[0,0],[0,0],[1,1]]))


def test_mne_topomap_masks_large_unmeasured_gap_and_does_not_invent_extrema():
    from experiments.scripts import render_tak_mismatch as renderer
    plt=renderer._plotting()
    xy=np.array([(x,y) for x in (-.75,-.55,.55,.75) for y in (-.2,0,.2)])
    values=1.+xy[:,1]+.2*xy[:,0]
    frame=pd.DataFrame(dict(x=xy[:,0],y=xy[:,1],display_value=values,pair_members=[str(i) for i in range(len(xy))]))
    fig,ax=plt.subplots()
    im=renderer.spatial_topomap(ax,frame,limits=(0,2),title="synthetic")
    field=im.get_array()
    x0,x1,y0,y1=im.get_extent()
    xi=int(np.argmin(abs(np.linspace(x0,x1,field.shape[1]))))
    yi=int(np.argmin(abs(np.linspace(y0,y1,field.shape[0]))))
    assert field.mask[yi,xi]
    assert field.count()>0
    assert field.min()>=values.min()-1e-8 and field.max()<=values.max()+1e-8
    plt.close(fig)


def test_full_head_preview_fills_gaps_and_border_without_inventing_extrema():
    from experiments.scripts import render_tak_mismatch as renderer
    plt = renderer._plotting()
    xy = np.array([(x,y) for x in (-.75,-.55,.55,.75) for y in (-.2,0,.2)])
    values = 1.+xy[:,1]+.2*xy[:,0]
    frame = pd.DataFrame(dict(x=xy[:,0],y=xy[:,1],display_value=values,
                             pair_members=[str(i) for i in range(len(xy))]))
    before = frame.copy(deep=True)
    fig, ax = plt.subplots()
    im = renderer.spatial_topomap(ax,frame,limits=(0,2),coverage="head")
    field = im.get_array()
    x0,x1,y0,y1 = im.get_extent()
    xx,yy = np.meshgrid(np.linspace(x0,x1,field.shape[1]),np.linspace(y0,y1,field.shape[0]))
    interior = xx*xx+yy*yy < .99**2
    assert not np.ma.getmaskarray(field)[interior].any()
    assert np.ma.getmaskarray(field)[xx*xx+yy*yy > 1.01].all()
    assert field.min() >= values.min()-1e-8 and field.max() <= values.max()+1e-8
    pd.testing.assert_frame_equal(frame,before)
    plt.close(fig)


def test_preview_reprojection_preserves_values_and_rejects_record_geometry_changes():
    import copy
    from experiments.scripts import render_tak_mismatch as renderer
    xyz = [(-.6,0.,.8),(.6,0.,.8),(0.,.8,.6),(0.,-.8,.6)]
    geometry = [dict(x=x,y=y,z=z,component=component,coordinate_system="dataset_mnt_pos_3d")
                for x,y,z in xyz for component in ("HbO","HbR")]
    pairs = [dict(id=f"CH{i+1}",component_indices=[2*i,2*i+1]) for i in range(4)]
    plan = dict(records=[dict(id=record,dataset="synthetic",subject="s1",pairs=pairs,
                             geometry_identity=dict(fnirs=copy.deepcopy(geometry))) for record in ("r1","r2")])
    frame = pd.DataFrame(dict(dataset="synthetic",subject="s1",period="subject_mean",
        record_ids="r1;r2",pair_members=[p["id"] for p in pairs],
        x=[p[0] for p in xyz],y=[p[1] for p in xyz],display_value=[1.,2.,np.nan,4.]))
    before = frame.copy(deep=True)
    result = renderer._spatial_preview_coordinates(frame,plan)
    pd.testing.assert_frame_equal(frame,before)
    np.testing.assert_array_equal(result.display_value,frame.display_value)
    np.testing.assert_allclose(result[["x_orthographic","y_orthographic"]],frame[["x","y"]])
    np.testing.assert_allclose(result.loc[2,["x","y"]].to_numpy(float),[0,.9*np.arctan2(.8,.6)/(np.pi/2)])
    for component in plan["records"][1]["geometry_identity"]["fnirs"][:2]:
        component["z"] = .7
    with pytest.raises(ValueError,match="record-varying"):
        renderer._spatial_preview_coordinates(frame,plan)


def test_region_review_pairs_subjects_and_keeps_incomplete_region_in_denominator():
    rows = []
    for subject, values in (("a",[.1,.1,.2,.2,.5,.5]),("b",[.2,.2,.4,.4,.3,.3]),
                            ("missing",[.1,.1,.2,.2,.5,np.nan])):
        for period, record in (("subject_mean","all"),("full","r1"),("full","r2")):
            for i,(pair,value) in enumerate(zip(["AF1","AF2","C3","C4","P3","P4"],values)):
                rows.append(dict(dataset="simultaneous_eeg_nirs",subject=subject,record_unit=record,
                    period=period,task_label="task",metric="D1",location_id=str(i),pair_members=pair,
                    x=i/10,y=i/20,display_value=value))
    result = analysis.spatial_region_patterns(pd.DataFrame(rows))
    s = result["regional_contrast_summary"].query("level == 'subject' and contrast == 'posterior_minus_central'").iloc[0]
    assert s.planned_subjects == 3 and s.available_subjects == 2 and s.positive_subjects == 1
    assert s['median'] == pytest.approx(.1)
    task = result["regional_contrast_summary"].query("level == 'task' and contrast == 'posterior_minus_central'").iloc[0]
    assert task.planned_subjects == 3 and task.available_subjects == 2  # six records are not six subjects
    missing = result["regional_values"].query("subject == 'missing' and region == 'posterior'")
    assert missing.value.isna().all() and missing.available_locations.eq(1).all()


def test_parameter_review_counts_groups_preserves_failures_and_uses_paired_rms():
    groups, windows = [], []
    for identity, size, tau, success in (("a",2,3.,True),("b",4,12.,False)):
        metadata = dict(group_id=identity,dataset="synthetic",subject="s1",record_id=identity,
                        task_label="task",pair="AF1",status="completed" if success else "partial")
        groups.append(dict(metadata,planned_windows=size))
        for i in range(size):
            windows.append(dict(metadata,id=f"{identity}:{i}",hb_only_success=success,joint_success=True,
                hb_only_kappa=.64,hb_only_tau=tau,joint_kappa=.3,joint_tau=5.,
                D1=1.+2*i,D1_free=2.+2*i if success else np.nan,
                D_EH=3.,D_EH_free=4. if success else np.nan,D4=1.,D4_free=.8))
    frame = pd.DataFrame(windows)
    result = analysis.shared_parameter_patterns(frame,pd.DataFrame(groups))
    parameters = result['parameter_groups']
    assert len(parameters) == 4
    bad = parameters.query("group_id == 'b' and view == 'hb_only'").iloc[0]
    assert not bad.valid_estimate and bad.tau == 12 and bad.tau_at_upper_bound
    site = result['parameter_site_summary'].query("view == 'hb_only'").iloc[0]
    assert site.valid_groups == 1 and site.planned_groups == 2 and site.boundary_valid_groups == 0
    paired = result['fit_pair_groups'].query("group_id == 'a' and metric == 'D1'").iloc[0]
    assert paired.fixed == pytest.approx(np.sqrt(5)) and paired.free == pytest.approx(np.sqrt(10))
    frame.loc[0,'hb_only_tau'] = 4.
    with pytest.raises(ValueError,match="changes within"):
        analysis.shared_parameter_patterns(frame,pd.DataFrame(groups))


def test_four_dataset_review_weights_records_not_the_number_of_repeated_windows():
    from src.data.unified_physiology import VISUAL_CHANNEL_IDENTITY_SCHEMA
    datasets = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation', 'refed')
    maps, subjects, shapes, groups, windows = [], [], [], [], []
    for dataset in datasets:
        raw, native, geometry = _fixture(dataset=dataset)
        raw['D1'] *= 1 + .001*raw.pair.str.removeprefix('CH').astype(int)
        native = analysis.aggregate_fixed_metrics(raw, metrics=('D1',))['native_pairs']
        geometry['native_eeg_identity'] = VISUAL_CHANNEL_IDENTITY_SCHEMA
        _, mm = analysis.aggregate_spatial_maps(native, analysis.spatial_half_metrics(raw), geometry)
        maps.append(mm)
        subjects.append(dict(dataset=dataset, subject='clear', metric='D1', value=2.))
        shapes.append(dict(dataset=dataset, subject='clear', feature='Hb_common_direction_fraction', value=.9))
        for record, size, tau in (('one', 2, 2.), ('two', 4, 8.)):
            identity = dataset+'|'+record
            meta = dict(group_id=identity, dataset=dataset, subject='clear', record_id=identity,
                        task_label='task', pair='AF1', status='completed')
            groups.append(dict(meta, planned_windows=size))
            for i in range(size):
                windows.append(dict(meta, id=f'{identity}|{i}', hb_only_success=True, joint_success=True,
                    hb_only_kappa=.64, hb_only_tau=tau, joint_kappa=.5, joint_tau=5.,
                    D1=1.+2*i, D1_free=2.+2*i, D_EH=3., D_EH_free=4., D4=1., D4_free=.8))
    result = analysis.four_dataset_pattern_tables(pd.DataFrame(subjects), pd.DataFrame(shapes),
        pd.concat(maps), pd.DataFrame(windows), pd.DataFrame(groups))
    fit = result['fit_subjects'].query("metric == 'D1'")
    np.testing.assert_allclose(fit.fixed, np.sqrt(13.))
    np.testing.assert_allclose(fit.free, np.sqrt(20.))
    assert fit.paired_windows.eq(6).all() and fit.available_records.eq(2).all()
    parameter = result['parameter_subjects'].query("view == 'hb_only'")
    np.testing.assert_allclose(parameter.tau, 4.)
    stability = result['spatial_subjects'].query("endpoint == 'early_late_rho'")
    np.testing.assert_allclose(stability.value, 1.)
    assert result['subject_summary'].available_subjects.eq(1).all()
    with pytest.raises(ValueError, match='all four'):
        analysis.four_dataset_pattern_tables(pd.DataFrame(subjects).iloc[:3], pd.DataFrame(shapes),
            pd.concat(maps), pd.DataFrame(windows), pd.DataFrame(groups))


def test_fixed_population_patterns_remove_common_template_and_keep_missing_people():
    rows=[]
    names=[f'{prefix}{i}' for prefix in ('AF','C','P') for i in range(4)]
    for s in range(5):
        for record in ('a','b'):
            for period in ('full','early','late'):
                for i,name in enumerate(names):
                    # Repeatable subject pattern on top of a stronger shared gradient.
                    value=10+i+.2*np.sin(i+s)
                    valid=not (s==4 and record=='b' and i<7)
                    rows.append(dict(dataset='simultaneous_eeg_nirs',subject=f's{s}',
                        record_unit=f'simultaneous_eeg_nirs|s{s}|{record}',task_label=record,period=period,
                        metric='D1',location_id=str(i),pair_members=name,x=i/10,y=i/20,
                        display_value=value if valid else np.nan,display_valid=valid,selection_valid=valid))
        for i,name in enumerate(names):
            rows.append(dict(dataset='simultaneous_eeg_nirs',subject=f's{s}',record_unit='all',task_label='all',
                period='subject_mean',metric='D1',location_id=str(i),pair_members=name,x=i/10,y=i/20,
                display_value=10+i+.2*np.sin(i+s),display_valid=True,selection_valid=True))
    frame=pd.DataFrame(rows)
    result=analysis.fixed_population_patterns(frame,bootstrap_samples=10,permutation_samples=20)
    good=result['spatial_repeat_subjects'].query("subject != 's4'")
    np.testing.assert_allclose(good[good.endpoint.eq('raw_rho')].value,1.)
    np.testing.assert_allclose(good[good.endpoint.eq('hotspot_overlap')].value,1.)
    np.testing.assert_allclose(good[good.endpoint.eq('hotspot_excess')].value,.75)
    # Independent scalar check: missing donors change the two population
    # medians, so even identical raw maps need not have centered rho exactly 1.
    from scipy.stats import spearmanr
    for s in range(4):
        residuals=[]
        for record in ('a','b'):
            residuals.append([.2*np.sin(i+s)-np.median([
                .2*np.sin(i+other) for other in range(5)
                if other!=s and not (other==4 and record=='b' and i<7)]) for i in range(12)])
        actual=good[good.subject.eq(f's{s}')&good.endpoint.eq('centered_rho')].value.iloc[0]
        assert actual==pytest.approx(spearmanr(*residuals).statistic)
    missing=result['spatial_repeat_subjects'].query("subject == 's4'")
    assert missing.value.isna().all() and missing.available_record_pairs.eq(0).all()
    summary=result['spatial_repeat_summary']
    assert summary.planned_subjects.eq(5).all() and summary.subjects.eq(4).all()
    hot=result['hotspot_subjects']
    assert hot[hot.subject.ne('s4')].stable_locations.eq(3).all()
    assert hot[hot.subject.eq('s4')].eligible_locations.eq(0).all()
    with pytest.raises(ValueError,match='Duplicate'):
        analysis.fixed_population_patterns(pd.concat([frame,frame.iloc[:1]]))
    common=frame.copy()
    common['display_value']=10+common.location_id.astype(float)
    common['display_valid']=True;common['selection_valid']=True
    only_template=analysis.fixed_population_patterns(common,bootstrap_samples=10,permutation_samples=20)
    fixed=only_template['spatial_repeat_subjects']
    np.testing.assert_allclose(fixed[fixed.endpoint.eq('raw_rho')].value,1.)
    np.testing.assert_allclose(fixed[fixed.endpoint.eq('hotspot_overlap')].value,1.)
    assert fixed[fixed.endpoint.eq('centered_rho')].value.isna().all()


@pytest.mark.parametrize('failure', ['reversed_map', 'missing_record'])
def test_clean_hb_atlas_requires_every_record_to_support_selected_example(failure):
    from experiments.scripts import render_tak_mismatch as renderer
    rows=[]
    base=np.arange(1.,13.)
    for subject in ('unsupported','supported'):
        second=base*1.1
        if subject=='unsupported':
            second=base[::-1] if failure=='reversed_map' else np.full(12,np.nan)
        for record,values in [('a',base),('b',second)]:
            for i,value in enumerate(values):
                rows.append(dict(dataset='d',subject=subject,period='full',metric='D1',
                    record_unit=record,location_id=str(i),pair_members=f'p{i}',
                    display_value=value,display_valid=np.isfinite(value)))
        average=np.sqrt(np.nanmean(np.array([base,second])**2,axis=0))
        for i,value in enumerate(average):
            rows.append(dict(dataset='d',subject=subject,period='subject_mean',metric='D1',
                record_unit='subject_mean',location_id=str(i),pair_members=f'p{i}',
                display_value=value,display_valid=True))
    frame=pd.DataFrame(rows);before=frame.copy(deep=True)
    ranks=pd.DataFrame(dict(dataset=['d','d'],subject=['unsupported','supported'],rank=[1,2]))
    selected,candidates,checks=renderer._hb_atlas_subject_selection(frame,ranks)
    assert selected.subject.tolist()==['supported']
    assert not candidates[candidates.subject.eq('unsupported')].eligible.any()
    assert selected.minimum_record_correlation.iloc[0]==pytest.approx(1.)
    assert len(checks)==2 and checks.peak_above_spatial_mean.gt(0).all()
    pd.testing.assert_frame_equal(frame,before)
