"""Array-only checks for mixed-rate observations; no local evidence is needed."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts.evaluate_eeg_proxy_rate import (
    ARMS, DEFAULT, operators, pack_observations, project_power, read_config,
    paired_summary,
)
from src.inference.observation_baselines import native_feature_operators


def test_config_rejects_unregistered_access(tmp_path):
    cfg = read_config(DEFAULT)
    cfg['source_run'] = 'data/protected'
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='boundary'):
        read_config(path)


def test_original_operator_is_exact_parent():
    op = native_feature_operators(120)
    old = np.zeros((360,360))
    old[0::3,0::3] = op['eeg']
    old[1::3,1::3] = old[2::3,2::3] = op['fnirs']@op['native_interpolation']
    mean, common = operators('A4')
    np.testing.assert_array_equal(mean, old)
    np.testing.assert_array_equal(common, old)


def test_high_rate_arms_share_hb_operator_and_common_target_map():
    b, bc = operators('B4_grid10')
    c, cc = operators('C10')
    np.testing.assert_array_equal(b[1::3],c[1::3])
    np.testing.assert_array_equal(b[2::3],c[2::3])
    np.testing.assert_array_equal(bc,cc)
    np.testing.assert_allclose(c@np.ones(900),0,atol=1e-12)
    assert not b[360:].any()  # packed 4 Hz observations occupy only 120 rows/modality


@pytest.mark.parametrize('arm',list(ARMS))
def test_observation_counts_and_time_weighted_loss(arm):
    target = np.zeros((120,3))
    eeg10 = np.zeros(300)
    y, mask, scale = pack_observations(target,eeg10,np.ones(3),arm,'full')
    ne = ARMS[arm]['eeg_rate_hz']*30
    assert mask.sum() == ne+240
    # A unit EEG residual sustained for the same 30 seconds has identical cost.
    assert np.sum(1/scale[mask[:,0],0]**2) == pytest.approx(120)
    assert np.sum(1/scale[mask[:,1],1]**2) == pytest.approx(120)
    assert np.isnan(y[~mask]).all()


def test_hidden_hb_cannot_enter_fit_inputs():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(120,3))
    changed = target.copy()
    changed[:,1:] += 100*rng.normal(size=(120,2))
    eeg10 = rng.normal(size=300)
    for arm in ARMS:
        first = pack_observations(target,eeg10,np.ones(3),arm,'Hb_hidden')
        second = pack_observations(changed,eeg10,np.ones(3),arm,'Hb_hidden')
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)


def test_native_recomputation_changes_power_window_and_preserves_reference():
    clock = np.arange(6000)/200
    envelope = 1+.7*np.sin(2*np.pi*1.3*clock)
    signal = np.column_stack([envelope*np.sin(2*np.pi*f*clock) for f in (3,6,10,15,22,36)])
    coord = dict(pc=np.ones(30)/np.sqrt(30),eeg_factor=1.)
    bands=read_config(DEFAULT)['tensor']['eeg_bands_hz']
    a, _ = project_power(signal,200,coord,4,bands)
    b, _ = project_power(signal,200,coord,10,bands)
    assert a.shape == (120,) and b.shape == (300,)
    assert a[:20].mean() == pytest.approx(0,abs=1e-12)
    assert b[:50].mean() == pytest.approx(0,abs=1e-12)
    assert not np.allclose(b,np.interp(np.arange(300)/10,np.arange(120)/4,a))


def test_paired_aggregation_equal_subject_weight_and_failures():
    rows=[]
    for subject,count,error in [('s1',1,1.),('s2',3,9.)]:
        for i in range(count):
            for arm in ('A4','C10'):
                value=error*(.25 if arm=='C10' else 1.)
                rows.append(dict(subject=subject,identity=f'{subject}_{i}',arm=arm,converged=True,
                                 nmse_EEG=value,nmse_HbO=value,nmse_HbR=value))
    result=paired_summary(pd.DataFrame(rows),'A4','C10',np.random.default_rng(8),100)
    np.testing.assert_allclose(result['reference_nrmse'],np.sqrt(5.))
    np.testing.assert_allclose(result['delta_nrmse'],-np.sqrt(5.)/2)
    rows[-1]['converged']=False
    result=paired_summary(pd.DataFrame(rows),'A4','C10',np.random.default_rng(8),100)
    assert result['planned_pairs']==4 and result['available_pairs']==3
    assert result['candidate_success']==3 and result['reference_success']==4
