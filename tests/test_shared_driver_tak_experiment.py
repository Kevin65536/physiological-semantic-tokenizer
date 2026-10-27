"""Array-free public scope and synthetic controls for Tak migration."""
from copy import deepcopy
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


CONFIG = Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_v1.yaml'


def test_tak_contract_and_complete_synthetic_matrix():
    cfg=runner.tak_config(CONFIG)
    assert json.loads(json.dumps(cfg)) == cfg
    specs=runner.tak_specs(cfg,None,'synthetic')
    assert len(specs)==3*3*8
    assert len({s['key'] for s in specs})==len(specs)
    a,op,sd,truth=runner.tak_synthetic(cfg,1,'tak_nonrest')
    assert a.shape==(120,3) and op.shape==(360,360) and sd.shape==(3,)
    np.testing.assert_array_equal(truth['states'][:,3],truth['states'][:,4])
    np.testing.assert_allclose(a[:20].mean(axis=0),0,atol=1e-16)
    mask=runner.volume_drive_mask(120,'joint_blocked')
    assert not mask[-1].any() and not mask[40:60,1:].any() and not mask[80:100,1:].any()


def test_tak_configuration_rejects_raw_and_protected_input(tmp_path):
    cfg=runner.tak_config(CONFIG)
    for key,value in [('raw_reads',True),('protected_data','allowed'),('fnirs_input','native')]:
        bad=deepcopy(cfg);bad['data'][key]=value
        p=tmp_path/(key+'.yaml');p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError,match='cached-input'):
            runner.tak_config(p)


def test_tak_sharing_keeps_native_session_support_separate(monkeypatch):
    cfg=runner.tak_config(CONFIG)
    refs=[]
    for i in range(48):
        refs.append(dict(id=f'r{i}',join_key=f'public|s{i//2}|record',subject=f's{i//2}',
            dataset=cfg['data']['datasets'][i//12],anchor={'hb_channel':'A'},native_session=i%2,
            hb_start_s=100*(i%2),alignment_support={'fnirs':[100*(i%2),100*(i%2+1)]}))
    monkeypatch.setattr(runner,'volume_drive_plan',lambda *a:dict(records=refs))
    p=runner.tak_plan(cfg,Path('.'))
    assert len(p['groups'])==48
    # Same native record/session with non-overlapping supports shares physiology.
    refs[1]['native_session']=refs[0]['native_session']
    refs[1]['alignment_support']=refs[0]['alignment_support']
    p=runner.tak_plan(cfg,Path('.'))
    assert len(p['groups'])==47
    refs[1]['hb_start_s']=20
    with pytest.raises(ValueError,match='overlapping'):
        runner.tak_plan(cfg,Path('.'))


def test_sensitivity_keeps_same_records_and_declared_single_changes():
    cfg=runner.tak_config(CONFIG)
    plan=dict(groups=[dict(id='g0',refs=[dict(id='r',dataset='synthetic')])])
    jobs=runner.tak_specs(cfg,plan,'sensitivity')
    assert len(jobs)==6
    assert all(j['length_s']==90 and j['condition']=='joint_full' for j in jobs)
    assert all(j['arm']['fit']==['kappa','tau'] for j in jobs)
    assert all(j['arm']['initial']=='volume_tied' for j in jobs)


def test_tak_cli_check_only_cannot_start_measured_work(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(runner,'tak_main',lambda args:seen.append(args.tak_stage))
    args=['runner','--tak-comparison','--run-dir',str(tmp_path),'--tak-stage','measured']
    monkeypatch.setattr(sys,'argv',args+['--check-only'])
    runner.main()
    assert seen==['check']
    monkeypatch.setattr(sys,'argv',args+['--nonlinear-fit'])
    with pytest.raises(SystemExit) as exc:runner.main()
    assert exc.value.code==2 and seen==['check']


def test_default_cli_selects_tak_without_a_mode_flag(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(runner,'tak_main',lambda args:seen.append((args.config,args.tak_stage)))
    monkeypatch.setattr(sys,'argv',['runner','--run-dir',str(tmp_path)])
    runner.main()
    assert seen==[(runner.DEFAULT,'check')]
    assert runner.tak_config(runner.DEFAULT)['schema']=='shared_driver_tak_10hz_v2'


@pytest.mark.parametrize('module,arguments',[
    ('evaluate_ssm_output_support',['--run-dir','unused','--stage','plan']),
    ('evaluate_shared_driver_spatial',[]),
    ('run_functional_tokenizer',['--output','unused','--stage','plan']),
    ('evaluate_model_compatibility',['--output','unused','--stage','plan']),
])
def test_pre_tak_cli_requires_explicit_contract_before_work(module,arguments,monkeypatch):
    import importlib
    entry=importlib.import_module('experiments.scripts.'+module)
    monkeypatch.setattr(sys,'argv',[module,*arguments])
    with pytest.raises(SystemExit) as exc:entry.main()
    assert exc.value.code==2
