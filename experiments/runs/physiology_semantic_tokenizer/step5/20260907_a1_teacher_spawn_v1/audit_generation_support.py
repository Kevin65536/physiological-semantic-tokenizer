"""Same registered A1 generation streams; no extra independent replicates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments import evaluate_step5 as s
from concurrent.futures import ProcessPoolExecutor
import json,numpy as np


def audit_case(item):
    axis,rep,theta,cfg=item;generator_axis=axis if axis!='R' else 'G';failures=[]
    for stream in (1,2,*range(10,18)):
        seed=s.case_seed(cfg,axis,rep,stream)
        try:s.localization.generate_matched(cfg,generator_axis,theta,seed)
        except Exception as exc:
            row=dict(axis=axis,replicate=rep,truth=theta,stream=stream,seed=seed,error=f'{type(exc).__name__}: {exc}',refinements=[])
            for substeps in (2,8,32):
                c={**cfg,'model':{**cfg['model'],'rk4_substeps':substeps}}
                rng=np.random.default_rng(seed);z=rng.normal(size=6)*c['model']['initial_state_std'];raw=s.localization.raw_parameters(c,generator_axis,theta)
                saved=[z.copy()];error=None;step=0
                try:
                    for step in range(1,c['model']['steps']):
                        z=s.localization.independent_transition(z,raw,c)+rng.normal(size=6)*np.array(c['model']['process_std'])*np.sqrt(c['model']['dt']);saved.append(z.copy())
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                values=np.array(saved)
                row['refinements'].append(dict(substeps=substeps,error=error,stop_time_s=step*c['model']['dt'],minimum_saved_flow=float(np.exp(values[:,2]).min()),last_saved_transformed_state=values[-1]))
            failures.append(row)
    return dict(axis=axis,replicate=rep,attempted_streams=10,failed_streams=len(failures),failures=failures)


if __name__=='__main__':
    run=Path(__file__).resolve().parent;cfg=s.load_config(run/'resolved_config.yaml');cal=ROOT/'experiments/runs/physiology_semantic_tokenizer/step5/20260907_a0_calibration_v1'
    items=[]
    for axis in ('G','W','Z','R'):
        for rep in range(60):
            theta=0. if axis=='R' else json.loads((cal/f'case_{axis}_{rep:02d}.json').read_text())['truth']
            items.append((axis,rep,theta,cfg))
    with ProcessPoolExecutor(max_workers=12) as pool:rows=list(pool.map(audit_case,items))
    out=dict(interpretation='Generation support audit on exactly the registered A1 main/donor/template streams. Zero new statistical replicates. Substep refinement changes the discrete transition and is diagnostic only; no frozen failure is reclassified.',attempted_streams=sum(r['attempted_streams'] for r in rows),failed_streams=sum(r['failed_streams'] for r in rows),failures=[f for r in rows for f in r['failures']])
    s.write_json(run/'generation_support_audit.json',out)
    print({'attempted_streams':out['attempted_streams'],'failed_streams':out['failed_streams']},flush=True)
