"""Same observations/capacity, continuous truth and terminal-failure contracts."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_ssm_state_continuity as run
from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver, nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import BalloonParameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def configs():
    cfg = run.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_state_continuity_v1.yaml')
    return cfg,yaml.safe_load((ROOT/cfg['source_config']).read_text())


def test_breaks_remove_only_cross_boundary_driver_penalties():
    r = np.repeat([0.,.01,-.01],4)
    options = dict(sd=np.ones(3),starts=[dict(driver=r,initial_state=np.r_[0.,np.ones(4)])],
        penalty=1.,initial_penalty=0.,max_evaluations=1,return_jacobian=True,numerical_backend='numba')
    y = np.zeros((12,3))
    default = fit_nonlinear_shared_driver(y,BalloonParameters(),.25,**options)
    explicit = fit_nonlinear_shared_driver(y,BalloonParameters(),.25,curvature_breaks=(),**options)
    np.testing.assert_array_equal(default['residual_jacobian'],explicit['residual_jacobian'])
    broken = fit_nonlinear_shared_driver(y,BalloonParameters(),.25,curvature_breaks=[4,8],**options)
    assert broken['roughness'] == 0. and default['roughness'] > 0.
    np.testing.assert_array_equal(broken['prediction'],default['prediction'])
    changed = np.flatnonzero(np.any(broken['residual_jacobian']!=default['residual_jacobian'],axis=1))
    np.testing.assert_array_equal(changed,36+np.array([2,3,6,7]))
    for invalid in ([1],[8,4],[4,4],[4.0],[True],[4,6]):
        with pytest.raises(ValueError,match='curvature_breaks'):
            fit_nonlinear_shared_driver(y,BalloonParameters(),.25,curvature_breaks=invalid,**options)


def test_synthetic_truth_is_continuous_and_arms_use_identical_capacity(configs):
    cfg,base = configs
    case = run.synthetic_case(cfg,base,'evaluation',0,'nonrest','common_colored')
    local,basis,op,big_basis = run.operators()
    assert big_basis.shape == (1080,12)
    for j in range(3):
        np.testing.assert_array_equal(op[j*360:(j+1)*360,j*360:(j+1)*360],local)
        np.testing.assert_array_equal(big_basis[j*360:(j+1)*360,j*4:(j+1)*4],basis)
        replay = nonlinear_driver_forward(case['driver'][j*120:(j+1)*120],case['states'][j*120,1:],
            run.parameters(base),.25,substeps=8,derivative=False,numerical_backend='numba')
        np.testing.assert_allclose(replay['states'],case['states'][j*120:(j+1)*120],atol=1e-14,rtol=0)
    # The middle starts from a propagated nonrest state even if the entire
    # continuous generator initially starts at rest; it is never reset there.
    rest = run.synthetic_case(cfg,base,'evaluation',0,'rest','neural_only')
    assert np.max(abs(rest['states'][120,1:]-np.r_[0.,np.ones(4)])) > 1e-5
    np.testing.assert_array_equal(rest['driver'],case['driver'])
    assert run.visibility('Hb_hidden').sum() == 840
    assert run.visibility('center_Hb').sum() == 1048
    assert cfg['seed_streams']['training'] != cfg['seed_streams']['evaluation']


def test_hidden_target_cannot_change_chain_or_independent_fit(configs):
    cfg,base = configs
    case = run.synthetic_case(cfg,base,'pilot',9,'nonrest','common_colored')
    sd = np.array([.025,.01,.004])
    arms,free = run.fit_arms(cfg,base,case,sd,'Hb_hidden')
    contaminated = deepcopy(case)
    contaminated['target'][~run.visibility('Hb_hidden')] = np.nan
    others,other_free = run.fit_arms(cfg,base,contaminated,sd,'Hb_hidden')
    for arm in arms:
        assert arms[arm]['converged'] and others[arm]['converged']
        np.testing.assert_array_equal(arms[arm]['prediction'],others[arm]['prediction'])
        np.testing.assert_array_equal(arms[arm]['observation_component'][120:240],np.zeros((120,3)))
    np.testing.assert_array_equal(arms['C_free']['prediction'][120:240],free[1]['prediction'])
    np.testing.assert_array_equal(free[1]['prediction'],other_free[1]['prediction'])


def test_summary_retains_failures_and_does_not_count_alias_as_new_subject(configs,tmp_path):
    cfg,_ = configs
    cfg = deepcopy(cfg); cfg['evaluation_identities']=3
    for spec in run.plans(cfg):
        rows=[]
        for mode in cfg['modes']:
            for arm in ('C_free','C_chain','target_only'):
                converged = not (spec['identity']==0 and arm=='C_chain')
                rows.append(dict(**spec,mode=mode,arm=arm,converged=converged,status='completed' if converged else 'failed_numerical',
                    score_nrmse=.1,HbO_nrmse=.1,HbR_nrmse=.1,physical_error=.1,component_error=.1,driver_error=.1))
        run.write_json(tmp_path/'tasks'/f'{run.task_key(spec)}.json',dict(status='completed',rows=rows,profiles=[]))
    summary = run.summarize(cfg,tmp_path)
    assert summary['planned_rows']==108 and summary['task_failures']==0
    assert summary['numerical_failures']['C_chain']==12
    row = summary['paired'][0]
    assert row['planned']==3 and row['paired_success']==2 and row['independent_identities']==3
    assert row['chain_mean']==pytest.approx(33.4)
    assert not summary['decision']['failure_gate']
    assert run.read_json(tmp_path/'verification.json')['target_only_is_alias']
