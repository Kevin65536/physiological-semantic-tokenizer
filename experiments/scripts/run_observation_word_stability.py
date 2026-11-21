#!/usr/bin/env python3
"""Paired development refits of four-bit observation words under training population changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.observation_tokenizer_evaluation import hierarchy_weights, probe_attributes
from src.analysis.observation_word_stability import subject_agreement, summarize, training_draw, word_support
from src.data.observation_relation_inputs import digest, load_routers, replay
from src.data.observation_tokenizer_panel import DATASETS, G0_ROOT
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer
from src.tokenizers.rank_observation_words import RankObservationWords

CONFIGS = ROOT / 'experiments/configs/physiology_semantic_tokenizer'
CONFIG = CONFIGS / 'observation_word_stability_v1.yaml'
PANEL = CONFIGS / 'observation_relations_v1.yaml'
SUITE = ROOT / 'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
TEST = ROOT / 'tests/test_observation_word_stability.py'
SOURCES = [Path(__file__).resolve(), CONFIG, PANEL, TEST, *[ROOT / 'src' / path for path in (
    'analysis/observation_word_stability.py', 'analysis/observation_tokenizer_evaluation.py',
    'data/observation_relation_inputs.py', 'data/observation_tokenizer_panel.py',
    'tokenizers/rank_observation_words.py', 'tokenizers/factorized_observation.py')]]
# Linux fork shares the immutable, verified panel with workers without a second disk copy.
WORK = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, result):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda v: v.item()) + '\n')
    tmp.replace(path)


def configuration(path):
    cfg = yaml.safe_load(path.read_text())
    base = yaml.safe_load(PANEL.read_text())
    if (json.loads(json.dumps(cfg, allow_nan=False)) != cfg or cfg != yaml.safe_load(CONFIG.read_text()) or cfg['schema'] != 'observation_word_stability_v1'
            or ROOT / cfg['panel_contract'] != PANEL or base['splits'] != cfg['splits']
            or cfg['splits'] != ['train', 'validation']
            or cfg['forbidden'] != ['test', 'protected', 'raw', 'teacher', 'events']
            or base['test_access'] != 'forbidden' or base['protected_access'] != 'forbidden'
            or ROOT / base['parent'] != SUITE / '20261120_development_v3'
            or ROOT / base['release'] != SUITE / '20261120_release_v1'
            or ROOT / base['g0'] != G0_ROOT or cfg['arms'] != ['standard', 'rank']
            or cfg['shapes'] != {'eeg': [1, 400], 'hb': [2, 20]}
            or cfg['word_bits'] != 4 or cfg['seeds'] != [17, 73, 127]):
        raise ValueError('invalid public observation-word scope, dimensions or pairing')
    return cfg, base


def tasks(cfg):
    contexts = [('full', 0), *(('bootstrap', i) for i in range(cfg['subject_bootstraps'])),
                *(('omit', d) for d in DATASETS)]
    return [(m, condition, index, seed) for condition, index in contexts
            for seed in cfg['seeds'] for m in cfg['shapes']]


def fit_arm(modality, arm, x, seed):
    model = (RankObservationWords(modality, seed=seed) if arm == 'rank' else
             FactorizedObservationTokenizer(modality, detail_count=0, seed=seed))
    return model.fit(x, partition='train')


def bench_job(i):
    mod = ('hb', 'eeg')[i % 2]
    x = np.random.default_rng(425+i).normal(size=(16384, *( (2, 20) if mod == 'hb' else (1, 400)))).astype('f')
    for arm in ('standard', 'rank'):
        c = fit_arm(mod, arm, x, 17)
        ids = c.encode(x[:64], mode='prototype')['pattern']
        if ids.shape != (64,) or ids.dtype.kind not in 'iu' or c.rate('prototype')['bits_per_item'] != 4:
            raise ValueError('synthetic payload mismatch')
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def pool(workers):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'))


def preflight(out, cfg):
    if (out/'preflight.json').exists():
        raise FileExistsError('preflight already completed')
    write(out/'resolved_config.json', cfg)
    sources = {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
    for p in SOURCES:
        target = out/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    results = []
    for n in cfg['resources']['worker_candidates']:
        start = time.perf_counter()
        with pool(n) as workers:
            rss = list(workers.map(bench_job, range(16)))
        seconds = time.perf_counter() - start
        results.append(dict(workers=n, seconds=seconds, fits_per_s=32/seconds, worker_peak_kib=max(rss)))
    best = max(results, key=lambda r: r['fits_per_s'])
    write(out/'preflight.json', dict(status='passed', sources=sources, finished=now(), benchmark=results,
                                    selected=best, fits_planned=2*len(tasks(cfg)), measured_arrays_read=False))


def cell_path(out, case):
    return out/'cells'/('{}_{}_{}_{}.json'.format(*case))


def run_cell(case):
    mod, cond, idx, seed = case
    out, cfg, t = WORK['output'], WORK['config'], WORK[mod]
    draw = training_draw(t['train'], t['weights'], cond, idx, cfg)
    xt = t['xt'][draw]
    rows = []
    for arm in cfg['arms']:
        model = fit_arm(mod, arm, xt, seed)
        pred = model.encode(t['xv'], mode='prototype')['pattern']
        if cond == 'full' and seed == 17:
            train_ids = model.encode(t['xt'], mode='prototype')['pattern']
            if arm == 'standard' and (not np.array_equal(pred, t['val'].pattern)
                                      or not np.array_equal(train_ids, t['train'].pattern)):
                raise ValueError('same-draw baseline did not reproduce the frozen dictionary')
            if arm == 'rank':
                path = out/f'{mod}_rank.joblib'
                joblib.dump(model, path.with_suffix('.tmp'), compress=3)
                path.with_suffix('.tmp').replace(path)
                if not np.array_equal(joblib.load(path).encode(t['xv'], mode='prototype')['pattern'], pred):
                    raise ValueError('saved candidate did not reproduce IDs')
            t['reference'][arm] = pred
            names = {'pattern': train_ids, 'wave_kmeans': t['train'].blind.to_numpy()}
            test_names = {'pattern': pred, 'wave_kmeans': t['val'].blind.to_numpy()}
            probes = probe_attributes(t['train'], t['val'], t['xt'], t['xv'], names, test_names, mod, 16)
            probes.assign(arm=arm).to_csv(out/f'{mod}_{arm}_attributes.csv', index=False)
            coverage = pd.DataFrame(word_support(t['val'], pred))
            coverage.assign(modality=mod, arm=arm).to_csv(out/f'{mod}_{arm}_support.csv', index=False)
        groups = {k: ix for k, ix in t['groups'].items() if cond != 'omit' or k[0] == idx}
        for score in subject_agreement(groups, t['reference'][arm], pred):
            rows.append(dict(modality=mod, condition=cond, perturbation=idx, seed=seed, arm=arm, **score))
    stats = dict(case=case, training_subjects=len(t['train'].iloc[np.unique(draw)][['dataset','subject']].drop_duplicates()),
                 draw_sha256=hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest(), rows=rows)
    write(cell_path(out, case), stats)


def population(frames, waves, cfg, out):
    WORK.update(output=out, config=cfg)
    for mod, shape in cfg['shapes'].items():
        t, v = (frames[mod, s] for s in cfg['splits'])
        pops = []
        for split, f in [('train', t), ('validation', v)]:
            pop = set(zip(f.dataset, f.subject))
            if (len(pop) != cfg['subjects'][split] or set(f.dataset) != set(DATASETS)
                    or waves[mod, split].shape != (len(f), *shape)):
                raise ValueError('unexpected population or tensors')
            pops.append(pop)
        if pops[0] & pops[1] or len(v) != cfg['validation_patches'][mod]:
            raise ValueError('training overlap or changed validation denominator')
        WORK[mod] = dict(train=t, val=v, xt=waves[mod, 'train'], xv=waves[mod, 'validation'],
                         weights=hierarchy_weights(t), groups=v.groupby(['dataset','subject']).indices, reference={})


def screen(out, cfg, base):
    pre = read(out/'preflight.json')
    if (pre['status'] != 'passed' or pre['sources'] != {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
            or cfg != read(out/'resolved_config.json')):
        raise ValueError('source or configuration changed since synthetic preflight')
    if (out/'summary.json').exists():
        raise FileExistsError('completed run')
    def progress(stage, **fields):
        write(out/'controller.json', dict(status='running', stage=stage, updated=now(), pid=os.getpid(), **fields))
    started = time.perf_counter()
    cases = tasks(cfg)
    (out/'cells').mkdir(exist_ok=True)
    try:
        progress('replay')
        routers, blind = load_routers(ROOT/base['parent'], ROOT/base['release'], base['identity'])
        frames, waves, audit = replay(ROOT/base['parent'], base['identity'], routers, blind, cfg['resources']['io_workers'])
        population(frames, waves, cfg, out)
        write(out/'panel_replay.json', audit)
        # These two parent-process cells reproduce both frozen assignments and initialize read-only references.
        references = [case for case in cases if case[1] == 'full' and case[3] == 17]
        for case in references:
            run_cell(case)
        missing = [case for case in cases if not cell_path(out, case).is_file()]
        done = len(cases)-len(missing)
        start_refits = time.perf_counter()
        n = pre['selected']['workers']
        progress('refits', completed_fits=2*done, planned_fits=2*len(cases), workers=n)
        with pool(n) as executor:
            pending = {}
            todo = iter(missing)
            def submit():
                case = next(todo, None)
                if case is not None:
                    pending[executor.submit(run_cell, case)] = case
            for _ in range(min(len(missing), 2*n)):
                submit()
            while pending:
                future = next(as_completed(pending))
                case = pending.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f'failed paired cell {case}') from exc
                done += 1
                submit()
                if done % 5 == 0 or done == len(cases):
                    seconds = time.perf_counter()-start_refits
                    rate = (done - (len(cases)-len(missing)))/seconds
                    progress('refits', completed_fits=2*done, planned_fits=2*len(cases), workers=n,
                             fits_per_s=2*rate, eta_seconds=(len(cases)-done)/rate)
        scores = []
        for case in cases:
            result = read(cell_path(out, case))
            if tuple(result['case']) != case:
                raise ValueError('refit identity changed')
            expected = {k for k in WORK[case[0]]['groups'] if case[1] != 'omit' or k[0] == case[2]}
            expected_rows = {(arm, *k) for k in expected for arm in cfg['arms']}
            rows = result['rows']
            if (len(rows) != len(expected_rows) or {(r['arm'], r['dataset'], r['subject']) for r in rows} != expected_rows
                    or not all(np.isfinite(r['ari']) for r in rows)):
                raise ValueError('missing or invalid paired subjects')
            scores.extend(rows)
        table = pd.DataFrame(scores)
        table.to_csv(out/'subject_agreement.csv.gz', index=False)
        def load_tables(suffix):
            return pd.concat([pd.read_csv(out/f'{m}_{a}_{suffix}.csv') for m in cfg['shapes'] for a in cfg['arms']])
        findings = summarize(table, load_tables('attributes'), load_tables('support'), cfg)
        write(out/'summary.json', dict(status='completed', experiment_id=cfg['experiment_id'], fits=2*len(cases),
                                      paired_cells=len(cases), subjects=cfg['subjects'], validation_patches=cfg['validation_patches'],
                                      findings=findings, elapsed_s=time.perf_counter()-started, finished=now()))
        write(out/'controller.json', dict(status='completed', finished=now(), summary='summary.json', completed_fits=2*len(cases)))
    except BaseException:
        error = dict(status='failed', finished=now(), trace=traceback.format_exc())
        write(out/f'failure_{time.time_ns()}.json', error)
        write(out/'controller.json', error)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--stage', choices=['preflight', 'screen'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg, base = configuration(args.config)
    out = args.output.absolute()
    if out != out.resolve() or out.parent != SUITE or '_word_stability_v' not in out.name:
        raise ValueError('expected a fresh word_stability run in the observation suite')
    out.mkdir(exist_ok=True)
    with (out/'controller.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'preflight':
            preflight(out, cfg)
        else:
            screen(out, cfg, base)


if __name__ == '__main__':
    main()
