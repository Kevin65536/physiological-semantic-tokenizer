import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    'tak_cache_version_analysis',
    Path(__file__).resolve().parents[1] / 'experiments/scripts/analyze_tak_cache_versions.py',
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_known_frequency_scale_and_constant():
    t = np.arange(800) / 10
    x = np.sin(2*np.pi*.1*t)
    a, b = audit.signal_features(x), audit.signal_features(7*x+4)
    assert a['centroid'] == pytest.approx(.1, abs=1e-10)
    assert a['core_fraction'] == pytest.approx(1)
    assert a['fast_fraction'] == pytest.approx(0, abs=1e-12)
    assert b['sd']/a['sd'] == pytest.approx(7)
    for k in ['kurtosis','mobility','centroid','core_fraction']:
        assert a[k] == pytest.approx(b[k])
    assert np.isnan(audit.signal_features(np.ones(800))['centroid'])


def test_equal_records_not_pooled_windows():
    frame = pd.DataFrame({'record':['a']*99+['b'], 'error':[0]*99+[10]})
    assert audit.record_equal(frame)['error'] == 5


def test_amplitude_weighting_preserves_subject_record_window_mass():
    # Duplicating channels in a window or having many windows must not increase its parent's mass.
    rows=[('d','a','r1','w1',i) for i in range(20)]
    rows += [('d','a','r1','w2',0),('d','a','r2','w1',0),('d','b','r1','w1',0)]
    f=pd.DataFrame(rows,columns=['dataset','subject','record','window','native_index'])
    f['weight']=audit.amplitude_weights(f)
    np.testing.assert_allclose(f.groupby('subject').weight.sum(),[1,1])
    assert f[(f.subject=='a')&(f.record=='r1')].weight.sum()==pytest.approx(.5)
    assert f[(f.subject=='a')&(f.record=='r1')&(f.window=='w1')].weight.sum()==pytest.approx(.25)
    np.testing.assert_equal(audit.amplitude_weighted_quantile([0,1,100],[.1,.8,.1],[.05,.5,.95]),[0,1,100])


def test_paired_signflip_and_holm():
    import itertools
    signs=np.array(list(itertools.product([-1.,1.],repeat=4)))
    assert audit.paired_signflip(np.zeros(4),signs)==1
    # Four equal positive subject differences: only all-plus/all-minus reach the observed statistic.
    assert audit.paired_signflip(np.ones(4),signs)==pytest.approx(3/17)
    np.testing.assert_allclose(audit.holm_adjust([.01,.04,.03]),[.03,.06,.06])


def test_panel_deterministic_and_deduplicated():
    names = [f'w{w:05}__p{p:03}.json' for w in range(5) for p in range(7)]
    selected = audit.select_panel(list(reversed(names)))
    assert selected == [f'w{w:05}__p{p:03}.json' for w in [0,2,4] for p in [0,3,6]]
    assert audit.select_panel(['w00000__p000.json']) == ['w00000__p000.json']


@pytest.mark.parametrize('gain,extra', [(1.,0.),(.8,.1),(1.,5.),(-2.,1.)])
def test_residual_geometry_matches_independent_arrays(gain,extra):
    t=np.arange(1000)/100
    x=np.sin(2*np.pi*t)+3
    s=gain*np.sin(2*np.pi*t)+extra*np.cos(4*np.pi*t)-2
    o=x-s
    actual=audit.residual_geometry(s.std()/x.std(),np.corrcoef(x,s)[0,1])
    assert actual['o_sd_ratio']==pytest.approx(o.std()/x.std(),abs=1e-8)
    assert actual['variance_sum_ratio']==pytest.approx((s.var()+o.var())/x.var())
    if o.std()>1e-10:
        assert actual['correlation_so']==pytest.approx(np.corrcoef(s,o)[0,1])
    else:
        assert np.isnan(actual['correlation_so'])


def test_pair_rejects_changed_hb_and_scored_gap(tmp_path):
    t = np.arange(100)/10
    target = np.column_stack([np.sin(t),np.cos(t),np.sin(t+.5)])
    common = dict(target=target,time_s=t,fit_mask=np.ones((100,3),bool),
                  prediction_valid=np.ones((100,3),bool),prediction=.8*target)
    a,b=tmp_path/'a.npz',tmp_path/'b.npz'
    np.savez(a,**common);np.savez(b,**common)
    assert len(audit.spectral_pair(a,b)) == 6
    changed=target.copy();changed[:,1]+=1
    np.savez(b,**dict(common,target=changed))
    with pytest.raises(ValueError,match='Hb targets changed'):
        audit.spectral_pair(a,b)
    mask=common['fit_mask'].copy();mask[30]=False
    np.savez(a,**dict(common,fit_mask=mask));np.savez(b,**dict(common,fit_mask=mask))
    with pytest.raises(ValueError,match='Noncontiguous'):
        audit.spectral_pair(a,b)


def test_visual_mask_calibration_ignores_excluded_amplitude(tmp_path):
    import json
    rng=np.random.default_rng(42)
    hb=rng.normal(size=(900,24,2))*[2,.5]
    record='visual_cognitive_motivation/S06/session_Probe1'
    pairs=[dict(native_index=i,status='available',channel_sets={'nearest6':list(range(6))}) for i in range(24)]
    observed=[]
    for scale in [1.,1000.]:
        root=tmp_path/str(scale);run=root/'run';out=root/'out';(out/'subjects').mkdir(parents=True)
        path=run/'features'/record;path.mkdir(parents=True)
        x=hb.copy();x[:,:10]*=scale
        np.savez(path/'w00005.npz',hb=x,hb_valid=np.ones(24,bool),eeg_valid=np.ones(31,bool),time_s=np.arange(900)/10)
        sd=np.sqrt(x.reshape(-1,2).var(0).mean())
        (run/'coordinates.json').write_text(json.dumps({'individual':{'visual_cognitive_motivation__S06':dict(
            hb_factor=.025/sd,training_samples=900*24,sources=[dict(record_key=record,window_key='w00005')])}}))
        (run/'plan.json').write_text(json.dumps({'records':[dict(dataset='visual_cognitive_motivation',subject='S06',
            cache_key=record,record='session_Probe1',pairs=pairs,cache_windows=[dict(key='w00005')],
            native_eeg_channel_identity={'channel_names':[f'E{i}' for i in range(31)]})]}))
        header=root/'header.json';header.write_text(json.dumps([dict(subject='S06',cache_key=record,excluded_channels=list(range(1,11)))]))
        r=audit.visual_mask_subject((dict(run=str(run),export=str(out),headers=str(header)),'S06'))
        assert r['records'][0]['hb_usable']==14
        assert r['records'][0]['eeg_available']==31
        expected=.025/np.sqrt(hb[:,10:].reshape(-1,2).var(0).mean())
        assert r['groups']['usable']['factor']==pytest.approx(expected)
        observed.append(r)
    assert observed[1]['groups']['all']['factor']<observed[0]['groups']['all']['factor']/100
    assert observed[1]['groups']['usable']['factor']==observed[0]['groups']['usable']['factor']


def test_tokenizer_audit_cancellation_and_undefined_signal():
    from experiments.scripts.analyze_tak_cache_versions import cache_audit_components
    t=np.arange(900)/10.;x=np.column_stack([np.sin(t),np.sin(t),np.zeros(900)])
    s=np.column_stack([x[:,0],5*x[:,1],np.ones(900)])
    rows=cache_audit_components(x,s,np.ones(3))
    assert rows[0]['faithful_small_residual']==1 and rows[0]['o_sd_ratio']==0
    assert rows[1]['severe_cancellation']==1
    np.testing.assert_allclose([rows[1]['s_sd_ratio'],rows[1]['o_sd_ratio'],rows[1]['correlation_so']],[5,4,-1])
    assert not rows[2]['defined'] and np.isnan(rows[2]['s_sd_ratio'])


def test_tokenizer_overlap_restores_window_reference_and_retains_disagreement():
    from experiments.scripts.analyze_tak_cache_versions import cache_audit_overlap
    raw=np.column_stack([np.sin(np.arange(1500)/100),np.cos(np.arange(1500)/100)])
    def w(start,bias):
        x=raw[start:start+899];ref=x[:50].mean(0)
        return dict(start=start/10,x=x-ref,s=x-ref+bias,ref=ref,sd=np.ones(2),state=np.ones((899,5)))
    result=cache_audit_overlap(w(0,0),w(600,.3))
    assert result['seconds']==29.9 and result['target_max_abs']<1e-14
    np.testing.assert_allclose(result['subject_nrmse'],[.3,.3])
    assert cache_audit_overlap(w(0,0),dict(w(600,0),start=200)) is None


def test_tokenizer_audit_aggregation_does_not_weight_subject_by_number_of_windows():
    from experiments.scripts.analyze_tak_cache_versions import cache_audit_aggregate
    import pandas as pd
    frame=pd.DataFrame([dict(dataset='d',subject='a',record='r1',component='HbO',v=1.)]*10+[dict(dataset='d',subject='a',record='r2',component='HbO',v=3.),dict(dataset='d',subject='b',record='r1',component='HbO',v=8.)])
    got=cache_audit_aggregate(frame,['component'],['v'])
    assert got.set_index('subject').v.to_dict()=={'a':2.,'b':8.}


def test_tokenizer_join_check_restores_reference_and_applies_stored_scale_once(tmp_path,monkeypatch):
    from experiments.scripts import analyze_tak_cache_versions as audit
    from types import SimpleNamespace
    import json
    root=tmp_path/'run';out=tmp_path/'audit';folder=out/'records/visual/s/r';folder.mkdir(parents=True)
    feature=root/'features/visual/s/r';feature.mkdir(parents=True)
    raw=np.arange(900*4,dtype=float).reshape(900,2,2)/100;reference=raw[:50].mean(0)
    np.savez(feature/'w00000.npz',hb=raw-reference,hb_reference=reference,hb_valid=np.ones(2,bool))
    norm=tmp_path/'norm.json';norm.write_text(json.dumps(dict(factor=.25)))
    sample=dict(id='sample',fnirs=(raw[100:400]*.25).reshape(300,4).T,valid_mask={'fnirs':np.ones((4,300),bool)},fnirs_normalization={'source':str(norm)})
    class Samples(list):pass
    dataset=Samples([sample]);dataset.samples=[dict(hb_start_s=10.)]
    monkeypatch.setattr(audit,'_CACHE_AUDIT_INPUT',dataset,raising=False)
    r=dict(dataset='visual',subject='s',record='r',cache_key='visual/s/r',pairs=[{},{}],cache_windows=[dict(key='w00000',hb_start_s=0.)])
    cfg=dict(output=str(out),ssm=str(root),joins={'visual/s/r':[dict(sample_id='sample',status='available',window_key='w00000',hb_offset_s=10.)]})
    result=audit.cache_audit_join_record((cfg,r,[0],{}))
    assert result['joined']==1 and result['max_abs_error']<1e-12


def test_tokenizer_audit_correlation_is_invariant_to_small_measurement_units():
    from experiments.scripts.analyze_tak_cache_versions import cache_audit_components
    t=np.arange(900)/10.;x=np.tile(np.sin(t)[:,None],(1,3));s=.7*x+.2*np.cos(t)[:,None]
    a=cache_audit_components(x,s,np.ones(3));b=cache_audit_components(x*1e-9,s*1e-9,np.ones(3)*1e-9)
    for r,z in zip(a,b):
        for key in ['correlation_sx','correlation_so','s_sd_ratio','o_sd_ratio','local_nrmse','subject_nrmse']:
            np.testing.assert_allclose(r[key],z[key],rtol=1e-12)


def test_tokenizer_join_reports_relative_rounding_difference_without_signal_mismatch(tmp_path,monkeypatch):
    from experiments.scripts import analyze_tak_cache_versions as audit
    root=tmp_path/'run';out=tmp_path/'audit';(out/'records/d/s/r').mkdir(parents=True);feature=root/'features/d/s/r';feature.mkdir(parents=True)
    raw=np.arange(1000*4,dtype=float).reshape(1000,2,2);h=raw[1:901];ref=h[:50].mean(0)
    np.savez(feature/'w00000.npz',hb=h-ref,hb_reference=ref,hb_valid=np.ones(2,bool))
    start=10.15;absolute=round(start*1000/1000*10)
    sample=dict(id='s',fnirs=raw[absolute:absolute+20].reshape(20,4).T,valid_mask={'fnirs':np.ones((4,20),bool)},fnirs_normalization=None)
    class Samples(list):pass
    dataset=Samples([sample]);dataset.samples=[dict(hb_start_s=start)]
    monkeypatch.setattr(audit,'_CACHE_AUDIT_INPUT',dataset,raising=False)
    r=dict(dataset='d',subject='s',record='r',cache_key='d/s/r',pairs=[{},{}],cache_windows=[dict(key='w00000',hb_start_s=.1)])
    cfg=dict(output=str(out),ssm=str(root),joins={'d/s/r':[dict(sample_id='s',status='available',window_key='w00000',hb_offset_s=start-.1)]})
    result=audit.cache_audit_join_record((cfg,r,[0],{}))
    assert result['max_abs_error']==0 and result['naive_offset_mismatched_samples']==1
