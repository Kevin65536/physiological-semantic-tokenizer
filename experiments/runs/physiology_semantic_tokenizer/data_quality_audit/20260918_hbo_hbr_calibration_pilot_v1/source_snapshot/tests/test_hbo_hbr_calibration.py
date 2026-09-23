"""Data-independent contracts for the A/B calibration screen."""
from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from experiments.scripts.evaluate_hbo_hbr_calibration import (
    CONFIG, read_config, software_checks, generate_panel, fit_target,
    temporal_operator, noise_factor, forward, optical_replay, observation_matrix,
    information_inventory,
)


def test_nonlinear_derivatives_optical_replay_and_temporal_processing():
    result=software_checks(read_config(CONFIG))
    assert max(result['derivative_max_errors']) < 2e-6
    assert result['compiled_processing_error'] < 1e-10
    assert result['intensity_MBll_closure_error'] < 1e-12
    assert result['retained_temporal_rank'] == 59


def test_both_counterfactuals_and_independent_calibration_are_paired():
    cfg=read_config(CONFIG)
    a=generate_panel('white',1,1.,cfg)
    b=generate_panel('gain2_white',1,1.,cfg)
    c=generate_panel('gain2_white',1,4.,cfg)
    assert np.array_equal(a['clean'],b['clean'])
    assert np.array_equal(b['driver'],c['driver'])
    assert not np.allclose(b['clean'],c['clean'])
    assert np.array_equal(b['optical'],c['optical'])
    assert np.array_equal(b['calibration_covariance'],c['calibration_covariance'])
    assert not np.array_equal(b['optical'],b['matrix'])
    assert not np.array_equal(b['calibration_covariance'],b['covariance'])


def test_optical_mixing_and_nonrest_are_separate_physical_operations():
    cfg=read_config(CONFIG)
    p=generate_panel('dpf_mixing_nonrest',0,2.,cfg)
    assert p['physical'][0,3] != p['physical'][0,4]
    assert np.allclose(optical_replay(p['clean'],'dpf_mixing',cfg),
                       p['clean']@observation_matrix('dpf_mixing',cfg).T,atol=1e-13)


def test_noiseless_oracle_recovers_without_truth_initialization():
    cfg=read_config(CONFIG);p=generate_panel('dpf_mixing_nonrest',0,2.,cfg)
    op,u,_=temporal_operator()
    target=(u.T@op@p['clean']@p['matrix'].T).ravel()
    result=fit_target(target,p['matrix'],p['covariance'],cfg)
    assert result['status']=='completed'
    assert abs(result['x'][0]-2.) < .002
    assert abs(result['x'][1]-.35) < .002
    assert np.linalg.norm(result['driver']-p['driver'])/np.linalg.norm(p['driver']) < .002


def test_covariance_has_cross_hb_and_temporal_structure():
    cfg=read_config(CONFIG)
    white=noise_factor('white',cfg);shared=noise_factor('shared_slow',cfg)
    cw,cs=white@white.T,shared@shared.T
    assert np.max(abs(cw[::2,1::2])) < 1e-16
    assert np.max(abs(cs[::2,1::2])) > 1e-5
    assert np.linalg.eigvalsh(cs-cw).min() > -1e-12
    assert np.linalg.eigvalsh(cw).min() > 0


def test_wrong_tensor_contract_rejected_before_any_data_read(tmp_path):
    cfg=read_config(CONFIG);cfg['synthetic']['duration_s']=120
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='300 native'):
        read_config(path)


def test_inventory_uses_record_metadata_without_signal_arrays(tmp_path,monkeypatch):
    import experiments.scripts.evaluate_hbo_hbr_calibration as runner
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from types import SimpleNamespace
    (tmp_path/'parent.csv').write_text('dataset_id,subject,record\neeg_fnirs_single_trial,subject_01,session_01\n')
    m=dict(measurement=dict(fnirs_preprocessing_state=dict(canonical_unit='relative',native_contract={}),
            eeg_preprocessing_state={}),metadata={},native_sample_rate_hz=10,source_files=[dict(path='fixture')],
            array_shapes={'fnirs':[1200,2]},sample_rate_hz=10)
    record=SimpleNamespace(manifest=m,dataset_id='eeg_fnirs_single_trial',canonical_subject_id='subject_01',base_record_id='session_01')
    def init(self,*args,**kwargs):
        self.records_by_join_key={'eeg_fnirs_single_trial|subject_01|session_01':[record]}
    monkeypatch.setattr(CleanPhysiologyCacheIndex,'__init__',init)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k: pytest.fail('no array reads in inventory'))
    cfg=read_config(CONFIG);cfg['parent_inventory']='parent.csv'
    summary=information_inventory(cfg,tmp_path)
    assert summary['records']==1
    assert summary['independently_calibrated_record_count']==0
    assert json.loads((tmp_path/'record_evidence.json').read_text())[0]['join_key'].endswith('session_01')
