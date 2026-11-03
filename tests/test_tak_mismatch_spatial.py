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
