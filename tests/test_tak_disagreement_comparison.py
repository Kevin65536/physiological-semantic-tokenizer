"""Synthetic paired-support and inference checks; no retained artifacts required."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import tak_mismatch as analysis


def fixture_frames():
    rows=[];geometry=[];weak=[]
    dataset='simultaneous_eeg_nirs'
    names=[f'{prefix}{i}' for prefix in ('AF','C','P') for i in range(4)]
    for person in range(7):
        subject=f's{person}'
        for part in range(2):
            record=f'{dataset}|{subject}|part{part}'
            for site,name in enumerate(names):
                base=1+site/12+.2*np.sin(site+person)
                drive=base if part==0 else 3-base
                geometry.append(dict(dataset=dataset,subject=subject,record_id=record,pair=name,
                    location_id=str(site),x=site/10,y=site/20,geometry_status='available'))
                weak.append(dict(dataset=dataset,subject=subject,record_unit=record,
                    period='full',metric='D_bal_40s',location_id=str(site),display_value=base,
                    selection_valid=True))
                for window in range(4):
                    rows.append(dict(dataset=dataset,subject=subject,record_id=record,
                        native_support_id='support',pair=name,window_start_s=30*window,
                        task_label='same_task',record_unit=record,repeat_id=record,
                        repeat_kind='same_task_native_record',record_order=part,
                        parameter_mode='fixed',D1=base,D_drive_H=drive,D_EH=2*drive,
                        zero_NMSE=2+site/12,prediction_NMSE=1.2,template_NMSE=1.4,wrong_NMSE=1.5))
    return pd.DataFrame(rows),pd.DataFrame(geometry),pd.DataFrame(weak)


def test_common_window_mask_keeps_plans_and_ratio_of_aggregated_squares():
    windows,_,_=fixture_frames()
    data=windows.iloc[:4].copy()
    data['D1']=[1,2,3,4];data['D_drive_H']=[2,3,4,np.nan]
    data['zero_NMSE']=[1,4,9,16]
    matched,support=analysis.matched_disagreement_windows(data)
    assert len(matched)==4 and support.common_available.tolist()==[True,True,True,False]
    assert np.isnan(matched.D1.iloc[-1]) and support.D1_available.all()
    aggregate=analysis.aggregate_fixed_metrics(matched,metrics=analysis.DISAGREEMENT_METRICS,include_hb_ratio=False)['subjects'].set_index('metric')
    assert aggregate.loc['D1','value']==pytest.approx(np.sqrt(14/3))
    assert aggregate.loc[analysis.HB_RATIO,'value']==pytest.approx(1.)
    assert aggregate.loc[analysis.DRIVE_RATIO,'value']==pytest.approx(29/14)
    assert aggregate.planned_windows.eq(4).all() and aggregate.available_windows.eq(3).all()
    # A common coordinate-unit change must cancel against the squared target.
    scaled=matched.copy()
    scaled[['D1','D_drive_H']]*=13;scaled['zero_NMSE']*=13**2
    changed=analysis.aggregate_fixed_metrics(scaled,metrics=(analysis.HB_RATIO,analysis.DRIVE_RATIO),include_hb_ratio=False)['subjects'].set_index('metric')
    np.testing.assert_allclose(changed.value,aggregate.loc[changed.index,'value'])


def test_all_subject_comparison_has_paired_map_effects_and_missing_denominators():
    windows,geometry,weak=fixture_frames()
    # A failed replay cannot be scored as a zero or replaced with a Hb-only fit.
    absent=windows.subject.eq('s6')&windows.record_order.eq(1)&windows.window_start_s.eq(90)
    windows.loc[absent,'D_drive_H']=np.nan
    tables=analysis.disagreement_dataset_tables(windows,geometry,weak,bootstrap_samples=12,permutation_samples=32)
    pairs=tables['paired_stability_subjects']
    raw=pairs.query("contrast == 'drive_minus_Hb' and endpoint == 'raw_rho'")
    assert len(raw)==7 and raw.value.notna().sum()==6
    np.testing.assert_allclose(raw.value.dropna(),-2.)
    assert raw[raw.subject.eq('s6')].available_pairs.eq(0).all()
    # The same subject remains visible under the explicitly relaxed sensitivity.
    relaxed=tables['relaxed_spatial_repeat_subjects']
    assert relaxed.query("subject == 's6' and metric == 'D_drive_H' and endpoint == 'raw_rho'").value.notna().all()
    hs=tables['hotspot_subjects']
    assert hs[hs.subject.eq('s6')].eligible_locations.eq(0).all()
    assert tables['window_support'].shape[0]==len(windows)
    tests=tables['paired_stability_tests']
    assert tests.planned_subjects.eq(7).all()
    assert (tests.q_value.dropna()>=tests.p_value.dropna()-1e-12).all()
    assert tables['prediction_baseline_subjects'].subject.nunique()==7


def test_pairing_happens_before_subject_medians():
    rows=[]
    for metric,values in [('D1',[0.,1.,np.nan]),('D_drive_H',[np.nan,.5,1.])]:
        for i,value in enumerate(values):
            rows.append(dict(dataset='d',subject='s',metric=metric,record_a='a',record_b=str(i),
                raw_rho=value,centered_rho=value,hotspot_excess=value))
    pair=pd.DataFrame(rows)
    early=pair.rename(columns={'record_b':'record_unit','raw_rho':'value'})
    result=analysis.disagreement_paired_stability(dict(spatial_record_pairs=pair,early_late_records=early))
    actual=result.query("contrast == 'drive_minus_Hb' and endpoint == 'raw_rho'").iloc[0]
    assert actual.available_pairs==1 and actual.value==pytest.approx(-.5)


def test_disagreement_contract_drift_rejected_before_evidence_reads(tmp_path, monkeypatch):
    import yaml
    from experiments.scripts import evaluate_tak_mismatch as runner
    path=runner.ROOT/'experiments/configs/physiology_semantic_tokenizer/tak_disagreement_comparison_v1.yaml'
    cfg=runner.read_disagreement_config(path)
    cfg['statistics']['minimum_site_windows_fraction']=.1
    changed=tmp_path/'changed.yaml';changed.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runner,'read_json',lambda *a:pytest.fail('evidence read before contract check'))
    with pytest.raises(ValueError,match='contract drift'):
        runner.physiology_disagreement_comparison(changed,tmp_path/'output',1)
