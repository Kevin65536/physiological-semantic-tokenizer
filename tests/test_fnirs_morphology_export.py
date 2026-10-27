"""Retained-only Tak morphology export; all inputs are temporary synthetic fixtures."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


CONFIG = Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_v1.yaml'


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_saved_json_configuration_preserves_scientific_notation_numbers(tmp_path):
    expected = runner.tak_config(CONFIG)
    path = tmp_path/'resolved_config.json'
    save(path,expected)
    assert '1e-05' in path.read_text()
    actual = runner.tak_config(path)
    assert isinstance(actual['solver']['gradient_tolerance'],float)
    assert actual==expected==json.loads(path.read_text())


@pytest.fixture
def retained(tmp_path):
    cfg = runner.tak_config(CONFIG)
    parent = tmp_path/cfg['output_root']/'synthetic_retained_fixture'
    records, groups = [], []
    for dataset_index,dataset in enumerate(cfg['data']['datasets']):
        refs = [dict(id=f'd{dataset_index}_r{i}',dataset=dataset,subject=f's{dataset_index}',record='record0',
            partition='evaluation',duration_s=90,parent=cfg['parents'][0],hb_start_s=100*i,
            join_key=f'{dataset}|s{dataset_index}|record0',anchor=dict(hb_channel='A')) for i in range(12)]
        records.extend(refs)
        groups.append(dict(id=f'g{dataset_index:03d}',sharing_identity=[dataset,f's{dataset_index}'],refs=refs))
    plan = dict(schema='shared_driver_tak_plan_v1',selection='all_exact_parent_public_evaluation_records',
                records=records,groups=groups)
    jobs = runner.tak_specs(cfg,plan,'measured')
    save(parent/'resolved_config.json',cfg)
    save(parent/'plan.json',plan)
    save(parent/'measured_summary.json',dict(schema='shared_driver_tak_summary_v1',status='completed',phase='measured',
        planned_tasks=len(jobs),terminal_tasks=len(jobs),planned_rows=2304,terminal_rows=2304,missing=[]))
    for spec in jobs:
        rows = [dict(id=ref['id'],dataset=ref['dataset'],subject=ref['subject'],record=ref['record'],
            group=spec['group']['id'],length_s=spec['length_s'],condition=spec['condition'],phase='measured',
            arm=spec['arm']['name'],qualified=True,converged=True,fit_status='converged') for ref in spec['group']['refs']]
        # JSON row order need not equal the retained tensor's declared ref order.
        rows.reverse()
        if spec['condition']=='joint_blocked' and spec['arm']['name']=='tak_fit':
            rows[0].update(qualified=False,converged=False,fit_status='failed_convergence')
        path = parent/'tasks'/(spec['key']+'.json')
        save(path,dict(spec=spec,status='completed',rows=rows))
        if (spec['length_s']==90 and spec['condition'] in ('joint_full','joint_blocked')
                and spec['arm']['name'] in ('legacy_free','legacy_free_fit','tak_fit')):
            n = 360
            t = np.arange(n)*cfg['tensor']['dt_s']
            target = np.stack([np.column_stack((np.sin(t*.1),np.sin(t*.4)+i*.01,np.cos(t*.35))) for i in range(12)])
            sd = np.array([[1.,1.+i*.01,2.+i*.01] for i in range(12)])
            visible = np.array([runner.volume_drive_mask(n,spec['condition'])]*12)
            np.savez_compressed(path.with_suffix('.npz'),target=target,prediction=.8*target,sd=sd,visible=visible,
                # The export must not load unrelated costly inference arrays.
                operator=np.array({'must_not_be_loaded':True},dtype=object))
    return cfg,parent,plan,jobs


def test_export_is_read_only_preserves_identity_failures_and_hidden_segments(retained,monkeypatch):
    from src.metrics import fnirs_morphology as metrics
    cfg,parent,plan,jobs = retained
    original = metrics.fnirs_morphology_metrics
    supports = []

    def checked(target,prediction,**kwargs):
        mask = kwargs['score_mask']
        supports.append(mask.copy())
        assert mask.shape==(360,2) and not mask[-1].any()
        # The fixture's record-specific Hb offset identifies the tensor row.
        index = int(round(target[0,0]/.01))
        np.testing.assert_allclose(kwargs['training_sd'],[1.+index*.01,2.+index*.01])
        np.testing.assert_array_equal(kwargs['time_s'],np.arange(360)*.25)
        return original(target,prediction,**kwargs)

    monkeypatch.setattr(metrics,'fnirs_morphology_metrics',checked)
    for name in ('tak_fit_group','tak_prepare','tak_plan','tak_run','tak_summarize'):
        monkeypatch.setattr(runner,name,lambda *a,**kw:pytest.fail('export invoked fitting/preparation/parent writes'))
    missing = next(s for s in jobs if s['group']['id']=='g000' and s['length_s']==90
        and s['condition']=='joint_full' and s['arm']['name']=='tak_fit')
    (parent/'tasks'/(missing['key']+'.npz')).unlink()
    no_prediction = next(s for s in jobs if s['group']['id']=='g001' and s['length_s']==90
        and s['condition']=='joint_full' and s['arm']['name']=='legacy_free_fit')
    no_prediction_path = parent/'tasks'/(no_prediction['key']+'.npz')
    with np.load(no_prediction_path,allow_pickle=False) as archive:
        retained_arrays = {key:archive[key] for key in ('target','sd','visible')}
    np.savez_compressed(no_prediction_path,**retained_arrays)
    before = {p.relative_to(parent):(p.stat().st_mtime_ns,p.read_bytes()) for p in parent.rglob('*') if p.is_file()}
    output = parent/'morphology_v1'
    args = Namespace(config=parent/'resolved_config.json',project_root=parent.parents[4],run_dir=parent,tak_stage='morphology',
        morphology_output=output,morphology_settings=None)
    summary = runner.tak_main(args)
    after = {p.relative_to(parent):(p.stat().st_mtime_ns,p.read_bytes()) for p in parent.rglob('*')
        if p.is_file() and not p.is_relative_to(output)}
    assert after==before and not (parent/'controller.lock').exists()
    assert summary['refits']==summary['raw_reads']==summary['protected_reads']==0
    result = json.loads((output/'morphology.json').read_text())
    table = pd.read_csv(output/'morphology.csv')
    assert result['scope']=='retained_tak_waveform_atlas_90s'
    assert result['summary']['planned_curves']==result['summary']['terminal_curves']==288
    assert result['summary']['component_rows']==len(table)==576
    assert len(result['rows'])==288 and set(table.component)=={'HbO','HbR'}
    assert table.morphology_status.eq('missing_npz').sum()==24
    assert table.morphology_status.eq('missing_arrays:prediction').sum()==24
    assert table[~table.qualified].fit_status.eq('failed_convergence').all()
    assert len(table[~table.qualified])==8
    row = next(r for r in result['rows'] if r['id']=='d0_r11' and r['arm']=='legacy_free' and r['condition']=='joint_full')
    assert row['source_row_index']==0 and row['array_row_index']==11
    assert row['ref']==plan['records'][11]
    assert len(table.merge(table,on=['id','length_s','condition','arm','component']))==len(table)
    assert {'source_task','source_arrays','source_row_index','array_row_index','pair_status','within_segment_amplitude_ratio',
        'slope_direction_2s_target_slope_energy_weighted_unrecovered_rate',
        'pair_slope_relationship_5s_target_slope_product_weighted_unrecovered_rate'}<=set(table)
    assert len(supports)==264
    hidden = [s for s in supports if s.sum()==240]
    assert len(hidden)==144
    for mask in hidden:
        changes = np.diff(np.r_[False,mask[:,0],False].astype(int))
        assert list(np.flatnonzero(changes==1))==[40,80,160,200,280,320]
        assert np.all(np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)==20)
    evaluated_hidden = next(r for r in result['rows'] if r['condition']=='joint_blocked')
    assert evaluated_hidden['metrics']['components'][0]['segment_count']==6
    assert (output/'export_source.py').exists() and (output/'metrics_source.py').exists()
    with pytest.raises(ValueError,match='fresh direct child'):
        runner.tak_morphology_export(cfg,parent,output)


@pytest.mark.parametrize('tamper',['path_traversal','row_identity','task_spec','summary','resolved_config','symlink'])
def test_export_rejects_bad_metadata_before_any_array_read(retained,monkeypatch,tmp_path,tamper):
    cfg,parent,plan,jobs = retained
    spec = next(s for s in jobs if s['length_s']==90 and s['condition']=='joint_full' and s['arm']['name']=='legacy_free')
    metadata = parent/'tasks'/(spec['key']+'.json')
    if tamper=='path_traversal':
        plan['groups'][0]['id']='../../outside'
        save(parent/'plan.json',plan)
    elif tamper=='row_identity':
        value=json.loads(metadata.read_text());value['rows'][0]['subject']='different_subject';save(metadata,value)
    elif tamper=='task_spec':
        value=json.loads(metadata.read_text());value['spec']['group']['refs'].reverse();save(metadata,value)
    elif tamper=='summary':
        value=json.loads((parent/'measured_summary.json').read_text());value['terminal_rows']=2303
        save(parent/'measured_summary.json',value)
    elif tamper=='resolved_config':
        value=deepcopy(cfg);value['data']['raw_reads']=True;save(parent/'resolved_config.json',value)
    else:
        external=tmp_path/'outside.json';external.write_bytes(metadata.read_bytes())
        metadata.unlink();metadata.symlink_to(external)
    monkeypatch.setattr(runner.np,'load',lambda *a,**kw:pytest.fail('array read before metadata/scope validation'))
    with pytest.raises(ValueError,match='morphology'):
        runner.tak_morphology_export(cfg,parent,parent/'invalid_export')
    assert not (parent/'invalid_export').exists()


def test_export_rejects_unknown_settings_and_invalid_output_without_parent_writes(retained,monkeypatch,tmp_path):
    cfg,parent,_,_ = retained
    settings=tmp_path/'settings.json';settings.write_text('{"invented_setting": 1}')
    monkeypatch.setattr(runner.np,'load',lambda *a,**kw:pytest.fail('invalid settings reached array reads'))
    with pytest.raises(ValueError,match='unknown morphology settings'):
        runner.tak_morphology_export(cfg,parent,parent/'invalid_export',settings)
    with pytest.raises(ValueError,match='fresh direct child'):
        runner.tak_morphology_export(cfg,parent,tmp_path/'outside_export')
    assert not (parent/'invalid_export').exists() and not (parent/'controller.lock').exists()


@pytest.mark.parametrize('bad_array',['shape','mask'])
def test_export_rejects_array_contract_mismatch(retained,bad_array):
    cfg,parent,_,jobs = retained
    spec = next(s for s in jobs if s['length_s']==90 and s['condition']=='joint_full' and s['arm']['name']=='legacy_free')
    path = parent/'tasks'/(spec['key']+'.npz')
    with np.load(path,allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in ('target','prediction','sd','visible')}
    if bad_array=='shape':
        arrays['prediction'] = arrays['prediction'][:,:-1]
    else:
        arrays['visible'][:,20,1] = False
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='morphology retained (array shape|scoring mask) mismatch'):
        runner.tak_morphology_export(cfg,parent,parent/'invalid_export')
    assert not (parent/'invalid_export').exists()


def test_morphology_cli_dispatches_read_only_stage(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(runner,'tak_main',lambda args:seen.append((args.tak_stage,args.morphology_output,args.morphology_settings)))
    monkeypatch.setattr(sys,'argv',['runner','--config',str(CONFIG),'--run-dir',str(tmp_path),'--tak-stage','morphology',
        '--morphology-output',str(tmp_path/'morphology_v1'),'--morphology-settings',str(tmp_path/'settings.json')])
    runner.main()
    assert seen==[('morphology',tmp_path/'morphology_v1',tmp_path/'settings.json')]
    arguments = sys.argv.copy()
    for incompatible in ('--check-only','--pilot','--nonlinear-fit'):
        monkeypatch.setattr(sys,'argv',[*arguments,incompatible])
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code==2 and len(seen)==1
