"""Repeated-estimation planning and tensor checks never need measured artifacts."""
from types import SimpleNamespace
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


def test_repeatability_contract_has_current_clock_and_separate_estimation_targets():
    cfg,core=runner.tak_repeatability_config(None)
    assert cfg['data']['protected_data']=='forbidden'
    assert core['tensor']['dt_s']==.1
    assert core['solver']['observation_reference_dt_s']==.25
    assert core['arms'][1]['log_sd']==[.125,.125]
    assert core['arms'][1]['warm']=='tak_fixed'
    tasks=runner.tak_specs(core,None,'synthetic')
    assert len(tasks)==6 and all(t['length_s']==90 for t in tasks)
    assert all(t['arm']['fit'] in ([],['kappa','tau']) for t in tasks)


def test_nonoverlap_selection_checks_both_clocks_and_spreads_over_record():
    refs=[dict(eeg_start_s=float(i),hb_start_s=float(i),event_id=i) for i in range(0,1000,10)]
    selected=runner._tak_repeat_select(refs,3)
    # Twelve disjoint candidates: midpoint index 5.5 rounds deterministically to 6.
    assert [r['eeg_start_s'] for r in selected]==[0.,540.,990.]
    candidates=[dict(eeg_start_s=0.,hb_start_s=0.,event_id=0),
                dict(eeg_start_s=100.,hb_start_s=80.,event_id=1),
                dict(eeg_start_s=200.,hb_start_s=190.,event_id=2)]
    assert [r['event_id'] for r in runner._tak_repeat_select(candidates,3)]==[0,2]


def _record(ds='simultaneous_eeg_nirs',duration=400.):
    return SimpleNamespace(dataset_id=ds,canonical_subject_id='fixture',base_record_id='cnt_wg',
        join_key=ds+'|fixture|cnt_wg',manifest={'array_shapes':{'eeg':[round(duration*200),6],
        'fnirs':[round(duration*10),2]},'source_files':[]})


def _event(i,t,lo,hi,label='WG'):
    return dict(event_index=i,eeg_time_ms=t*1000,fnirs_time_ms=t*1000,label=label,
        metadata=dict(task='wg',alignment_support_ms={'eeg':[lo,hi],'fnirs':[lo,hi]}))


def test_disjoint_native_supports_remain_separate_parameter_targets():
    events=[_event(0,10,None,150000),_event(1,100,None,150000),_event(2,210,200000,None)]
    rows=runner._tak_repeat_candidates(_record(),events,{'hb_channel':'A'},coordinate='fixture__montage0')
    assert len(rows)==2  # the event at 100 s would cross the declared gap
    assert len({r['native_group_key'] for r in rows})==2
    assert [r['eeg_start_s'] for r in rows]==[5.,205.]
    assert all(r['coordinate']=='fixture__montage0' for r in rows)


@pytest.mark.parametrize('label,tier',[('LVLA (Sad)','primary'),('HVHA (Happy)','supplementary')])
def test_refed_conditions_are_separate_and_short_videos_are_not_stretched(label,tier):
    event=_event(0,0,None,None,label);event['metadata']['task']='emotion_video'
    rows=runner._tak_repeat_candidates(_record('refed',90),[event],{'hb_channel':'A'})
    assert len(rows)==1 and rows[0]['analysis_tier']==tier
    assert rows[0]['eeg_start_s']==0 and rows[0]['duration_s']==90
    assert runner._tak_repeat_candidates(_record('refed',89.9),[event],{'hb_channel':'A'})==[]


def test_preparation_stops_before_measured_access_if_synthetic_is_incomplete(tmp_path,monkeypatch):
    plan={'fixture':True};monkeypatch.setattr(runner,'tak_repeatability_plan',lambda *args:plan)
    (tmp_path/'synthetic_summary.json').write_text(json.dumps({'status':'incomplete'}))
    (tmp_path/'synthetic_rows.json').write_text('[]')
    import src.data.clean_physiology_cache as cache
    monkeypatch.setattr(cache,'CleanPhysiologyCacheIndex',lambda *args:pytest.fail('No measured index access expected'))
    with pytest.raises(ValueError,match='successful synthetic'):
        runner.tak_repeatability_prepare({}, {},tmp_path,tmp_path,plan,1)


def test_modified_repeatability_scope_is_rejected(tmp_path):
    import yaml
    cfg,_=runner.tak_repeatability_config(None);cfg=deepcopy(cfg)
    cfg['data']['subjects']='all_subjects'
    path=tmp_path/'altered.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='unchanged versioned scope'):
        runner.tak_repeatability_config(path)


def _statistics_rows():
    import pandas as pd
    rows=[]
    for subject,scale in [('s1',1.),('s2',2.),('s3',3.)]:
        for group,values in [('g1',[.8,1.2]),('g2',[1.4,1.6])]:
            for j,value in enumerate(values):
                rows.append(dict(id=f'{subject}_{group}_{j}',dataset='fixture',analysis_tier='primary',
                    subject=subject,native_group=subject+group,qualified=True,fixed_qualified=True,
                    kappa=scale*value,tau=2*scale*value,full_hb_nmse=.1,fixed_full_hb_nmse=.2))
    return pd.DataFrame(rows)


def test_summary_uses_subject_clusters_not_window_count_for_uncertainty():
    import pandas as pd
    frame=_statistics_rows()
    subjects,a,baseline=runner._tak_repeatability_statistics(frame,bootstrap_repeats=100)
    _,b,_=runner._tak_repeatability_statistics(pd.concat([frame]*4,ignore_index=True),bootstrap_repeats=100)
    fields=['between_subject_log_rms','within_subject_log_rms','between_within_ratio','ratio_ci95_low','ratio_ci95_high']
    np.testing.assert_allclose(a[fields],b[fields],atol=1e-14)
    assert a.included_subjects.eq(3).all()
    assert baseline.paired_windows.iloc[0]==12 and baseline.paired_subjects.iloc[0]==3


def test_native_groups_are_equal_weight_and_failed_values_do_not_enter_centers():
    import pandas as pd
    frame=_statistics_rows();first=frame[frame.subject.eq('s1')].copy()
    first.loc[first.native_group.eq('s1g1'),'kappa']=1.
    first.loc[first.native_group.eq('s1g2'),'kappa']=4.
    repeated=pd.concat([first,first[first.native_group.eq('s1g1')]]*2,ignore_index=True)
    failed=repeated.iloc[[0]].copy();failed['kappa']=1000.;failed['qualified']=False
    subject,_,_=runner._tak_repeatability_statistics(pd.concat([repeated,failed],ignore_index=True),bootstrap_repeats=10)
    k=subject[subject.parameter.eq('kappa')].iloc[0]
    assert k.geometric_center==pytest.approx(2.)
    assert k.planned_windows==len(repeated)+1 and k.qualified_windows==len(repeated)


def test_one_estimate_per_record_does_not_imply_zero_within_record_variation():
    frame=_statistics_rows().groupby(['subject','native_group'],sort=False).head(1)
    subject,_,_=runner._tak_repeatability_statistics(frame,bootstrap_repeats=10)
    assert subject.within_support_log_rms.isna().all()
    assert subject.total_within_subject_log_rms.notna().all()
    collapsed=_statistics_rows().assign(kappa=.64,tau=2.)
    _,summary,_=runner._tak_repeatability_statistics(collapsed,bootstrap_repeats=10)
    assert summary.between_subject_log_rms.eq(0).all()
    assert summary.between_within_ratio.isna().all()
