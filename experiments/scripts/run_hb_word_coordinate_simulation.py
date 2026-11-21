#!/usr/bin/env python3
"""Supervised synthetic Hb word ablation with per-population checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.hb_word_coordinate_simulation import evaluate_cell, summarize
from src.data.observation_relation_inputs import digest

CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/hb_word_coordinate_simulation_v1.yaml'
SUITE = ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
SOURCES = [Path(__file__).resolve(), CONFIG, ROOT/'tests/test_hb_word_coordinate_simulation.py',
           *[ROOT/'src'/p for p in ('analysis/hb_word_coordinate_simulation.py', 'tokenizers/partial_rank_hb_words.py',
                                    'tokenizers/rank_observation_words.py', 'tokenizers/factorized_observation.py')]]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(p)


def path_for(out, rep, regime):
    return out/'cells'/f'{rep:03d}_{regime}.json'


def run_one(out, cfg, case):
    rep, regime = case
    result = evaluate_cell(cfg, rep, regime)
    save(path_for(out,*case), result)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_pool(out, cfg, cases, count, callback):
    peak = 0
    with ProcessPoolExecutor(count, mp_context=multiprocessing.get_context('fork')) as pool:
        stream = iter(cases)
        pending = {}
        def submit_one():
            case = next(stream,None)
            if case is not None:
                pending[pool.submit(run_one,out,cfg,case)] = case
        for _ in range(min(len(cases),2*count)):
            submit_one()
        while pending:
            future = next(as_completed(pending))
            case = pending.pop(future)
            try:
                peak = max(peak,future.result())
            except Exception as exc:
                raise RuntimeError(f'failed generator cell {case}') from exc
            callback()
            submit_one()
    return peak


def run(out, cfg):
    t0 = time.monotonic()
    cases = [(rep, r) for rep in range(cfg['replicates']) for r in cfg['regimes']]
    per_cell = 2*len(cfg['arms'])*len(cfg['fitting_seeds'])
    out.joinpath('cells').mkdir(exist_ok=True)
    config_path = out/'resolved_config.json'
    sources = {str(p.relative_to(ROOT)):digest(p) for p in SOURCES}
    if config_path.exists():
        if cfg != json.loads(config_path.read_text()) or sources != json.loads((out/'source_manifest.json').read_text()):
            raise ValueError('configuration or source changed between attempts')
    else:
        save(out/'source_manifest.json',sources)
        save(config_path,cfg)
    remaining = [case for case in cases if not path_for(out,*case).exists()]
    done = len(cases)-len(remaining)
    initial_done = done
    stage = 'throughput'
    n = None
    def progress():
        elapsed = time.monotonic()-t0
        rate = (done-initial_done)/elapsed
        save(out/'controller.json',dict(status='running',stage=stage,updated=now(),pid=os.getpid(),workers=n,
                                       completed_fits=per_cell*done,planned_fits=per_cell*len(cases),
                                       eta_seconds=(len(cases)-done)/rate if rate else None))
    def completed():
        nonlocal done
        done += 1
        if done%8 == 0 or done==len(cases):
            progress()
    progress()
    try:
        bench = out/'resources.json'
        if not bench.exists():
            tests = []
            for n in cfg['resources']['worker_candidates']:
                batch = remaining[:cfg['resources']['benchmark_cells_per_arm']]
                if not batch:
                    break
                start = time.monotonic()
                peak = run_pool(out,cfg,batch,n,completed)
                del remaining[:len(batch)]
                tests.append(dict(workers=n, fits_per_s=len(batch)*per_cell/(time.monotonic()-start),
                                  cells=len(batch),worker_peak_kib=peak))
            # Only needed on interruption after every numerical cell but before the benchmark write.
            best = max(tests,key=lambda r:r['fits_per_s'])['workers'] if tests else 1
            save(bench,dict(timings=tests,selected_workers=best,reason='highest measured fit rate on distinct complete replicate-regime groups; one numerical thread per worker; same declared grid'))
        n = json.loads(bench.read_text())['selected_workers']
        stage = 'population_refits'
        progress()
        run_pool(out,cfg,remaining,n,completed)
        frame = pd.DataFrame([row for case in cases for row in json.loads(path_for(out,*case).read_text())])
        result = summarize(frame,cfg)
        frame.to_csv(out/'population_scores.csv.gz',index=False)
        save(out/'summary.json',dict(status='completed',experiment_id=cfg['experiment_id'],fits=len(cases)*per_cell,
                                    independent_replicates=cfg['replicates'],cells=len(cases),finished=now(),
                                    elapsed_s=time.monotonic()-t0,findings=result))
        save(out/'controller.json',dict(status='completed',finished=now(),completed_fits=len(cases)*per_cell,summary='summary.json'))
    except BaseException:
        result = dict(status='failed',finished=now(),completed_fits=done*per_cell,trace=traceback.format_exc())
        save(out/f'failure_{time.time_ns()}.json',result)
        save(out/'controller.json',result)
        raise


def configuration(path):
    cfg = yaml.safe_load(path.read_text())
    if (cfg != yaml.safe_load(CONFIG.read_text()) or cfg != json.loads(json.dumps(cfg)) or
            cfg['data'] != 'generated_in_memory_only' or cfg['modalities'] != {'hb':[2,20]} or
            cfg['k'] != 16 or cfg['word_bits'] != 4 or cfg['fitting_seeds'] != [17,73,127] or
            cfg['primary_candidate'] != 'rank_levels' or cfg['arms'] != ['standard','rank_levels','rank_slopes','rank_all']):
        raise ValueError('invalid synthetic contract')
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=CONFIG)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    cfg = configuration(args.config)
    out = args.output.absolute()
    if out != out.resolve() or out.parent != SUITE or '_hb_coordinate_sim_v' not in out.name:
        raise ValueError('expected a dedicated simulation run under the observation suite')
    out.mkdir(exist_ok=True)
    with (out/'controller.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists():
            raise FileExistsError('completed simulation')
        run(out,cfg)


if __name__ == '__main__':
    main()
