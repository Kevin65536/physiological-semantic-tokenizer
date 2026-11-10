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
