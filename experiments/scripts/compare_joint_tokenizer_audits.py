"""Apply one frozen numerical contract across retained, example-matched development score records."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from review_joint_tokenizer import score


def table(run, output):
    output.mkdir(parents=True,exist_ok=False)
    contract_path=Path(__file__).resolve().parents[1]/'configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml'
    contract=yaml.safe_load(contract_path.read_text())
    with np.load(run/'panels.npz') as f:
        panels={k:f[k] for k in f.files if k.endswith(('_screen','_development'))}
    references={}
    for role,tree in [('screen','baseline_audit'),('development','development_audit')]:
        for m in ('eeg','hb'):
            data=[]
            for model in ('vq128','continuous'):
                base=run/tree/m/model
                path=(base/'summary_v2.json') if (base/'summary_v2.json').is_file() else base/'summary.json'
                with np.load(base/'metrics.npz') as f:
                    np.testing.assert_array_equal(panels[m+'_'+role],f['examples'])
                data.append(json.loads(path.read_text()))
            references[m,role]=data
    rows=[]; skipped=[]
    def record(d, source, raw, arm, replay, eligible):
        if not raw.is_file():
            skipped.append((str(source.relative_to(run)),arm,'no matching metric archive')); return
        with np.load(raw,allow_pickle=False) as arch:
            if 'examples' not in arch:
                skipped.append((str(source.relative_to(run)),arm,'no example triples'));return
            ex=arch['examples']
        matches=[key for key,panel in panels.items() if ex.shape==panel.shape and np.array_equal(ex,panel)]
        if len(matches)!=1:
            skipped.append((str(source.relative_to(run)),arm,'nonidentical panel/split or dataset shard'));return
        m,role=matches[0].split('_')
        d=dict(d,role=role,modality=m,checkpoint=d.get('checkpoint',str(source)),integer_replay={'status':'passed' if replay else 'unverified'})
        res=score(d,*references[m,role],contract)
        rows.append(dict(source=str(source.relative_to(run)),arm=arm,role=role,modality=m,inputs=len(ex),
                         eligibility=eligible,integer_replay=replay,macro_nmse=d['selection_score'],
                         all_numeric=not res['failed'] and res['finite_predictions'],
                         reconstruction_accepted=bool(eligible and res['reconstruction_passed']),failed_checks='|'.join(res['failed']),
                         nmse=json.dumps(d['metrics']['nmse']['macro']),rms_gain=json.dumps(d['metrics']['rms_gain']['macro'])))
    # Named score exports only. Source teacher and calibration-control scores are independent diagnostics.
    for tree in ['candidate_audit','development_audit','baseline_audit']:
        for path in sorted((run/tree).rglob('summary.json')):
            raw=path.parent/'metrics.npz'
            if not raw.is_file():continue
            chosen=path.with_name('summary_v2.json') if path.with_name('summary_v2.json').exists() else path
            d=json.loads(chosen.read_text())
            if 'metrics' not in d or 'integer_replay' not in d:continue
            replay=d['integer_replay'].get('status')=='passed'
            eligible=replay and 'temporal_diagnostic' not in path.parts and 'dictionary_target_oracle_v1' not in path.parts
            record(d,chosen,raw,'primary',replay,eligible)
    for path in sorted(run.rglob('screen_summary.json')):
        obj=json.loads(path.read_text())
        if 'metrics' in obj and isinstance(obj.get('integer_replay'), dict):
            replay = obj['integer_replay'].get('status') == 'passed'
            record(obj, path, path.parent / 'metrics.npz', 'primary', replay,
                   replay and obj.get('interface_eligible', True))
        for arm,d in obj.get('summaries',{}).items():
            if not isinstance(d,dict) or 'metrics' not in d:continue
            ok=obj.get('standalone_metric_replay')=='exact_from_saved_ID_file_after_model_reload'
            if isinstance(obj.get('integer_replay'),dict):
                ok=obj['integer_replay'].get('status')=='passed'
            diag=arm in ('source_Wiener','continuous_slots') or obj.get('interface_eligible') is False
            if 'cross_branch_readout_v1' in path.parts and path.parent.name not in ('full','same_branch','full_layout'):diag=True
            record(d,path,path.parent/(arm+'_metrics.npz'),arm,ok and not diag,not diag)
    phase=run/'spectral'
    for path in sorted(phase.rglob('summary.json')):
        raw=path.parent/'metrics.npz'
        if not raw.is_file():continue
        d=json.loads(path.read_text())
        if 'integer_replay' in d and 'metrics' in d:
            ok=d['integer_replay'].get('status')=='passed'
            record(d,path,raw,'primary',ok,ok)
    with (output/'numeric_checks.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    result=dict(status='completed',contract=str(contract_path),matched_scoring_records=len(rows),
                accepted_reconstruction=[v for v in rows if v['reconstruction_accepted']],
                skipped=skipped,scope='matched score records, including repeated milestones and controls. This is only the reconstruction gate; no pattern or visual pass is inferred.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(rows),'score conditions;',len(result['accepted_reconstruction']),'eligible reconstruction passes;',len(skipped),'unmatched records')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();table(args.run,args.output)
