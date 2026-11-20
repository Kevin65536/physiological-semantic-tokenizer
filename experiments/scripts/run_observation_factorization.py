#!/usr/bin/env python3
"""A bounded observation-only development screen, not a replacement for S/O qualification."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.observation_tokenizer_evaluation import (
    KEYS, hierarchy_weights, paired_contrast, pattern_atlas, probe_attributes, reconstruction, roll_patterns)

CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/factorized_observation_v1.yaml'
RUNS = ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
G0 = 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261117_base_g0_v2'
TASK_CACHE = 'data/cache/physiology_downstream_v2'


def serial(x):
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, (np.ndarray, tuple)):
        return list(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(str(type(x)))


def write_json(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data, default=serial, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg != yaml.safe_load(CONFIG.read_text()):
        raise ValueError('use the reviewed observation screening contract')
    data, m = cfg['data'], cfg['model']
    if (cfg['experiment_id'] != 'TK-OBS-FACTOR-v1' or data['g0'] != G0 or data['task_cache'] != TASK_CACHE
            or data['splits'] != ['train', 'validation'] or data['test_access'] != 'forbidden'
            or data['protected_access'] != 'forbidden' or data['raw_reads'] is not False
            or m['shapes'] != {'eeg':[1,400], 'hb':[2,20]} or m['seeds'] != [17,73,127] or m['k'] != 16):
        raise ValueError('invalid source, split, shape or seed contract')
    return cfg


def source_identity():
    paths = [CONFIG, Path(__file__), *(ROOT/p for p in ['src/data/observation_tokenizer_panel.py',
        'src/tokenizers/factorized_observation.py', 'src/analysis/observation_tokenizer_evaluation.py',
        'src/analysis/observation_legacy_control.py', 'src/analysis/observation_task_probe.py'])]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def synthetic(cfg):
    from src.tokenizers.factorized_observation import FactorizedObservationTokenizer as Codec
    rng = np.random.default_rng(8017); checks = {}
    for mod, dims in cfg['model']['shapes'].items():
        x = rng.normal(size=(128, *dims)).astype('f'); z = rng.normal(size=(16, *dims)).astype('f')
        for routing in ('pattern', 'waveform'):
            c = Codec(mod, n_patterns=16, detail_count=2, detail_bits=5, seed=17, routing=routing).fit(x, partition='train')
            for mode, rate in [('full', 14), ('flat', 14), ('prototype', 4), ('flat_pattern', 4)]:
                payload = c.encode(z, mode=mode); pred = c.decode(payload, mode=mode)
                if (not all(np.issubdtype(a.dtype, np.integer) for a in payload.values()) or pred.shape != z.shape
                        or not np.isfinite(pred).all() or c.rate(mode)['bits_per_item'] != rate):
                    raise ValueError('integer-only synthetic check failed')
        checks[mod] = 'integer_shapes_rates_and_finite_outputs_passed'
    return dict(status='passed', checks=checks, read_measured_arrays=False, sources=source_identity())


def read_panel(cfg, limits, destination, workers):
    from src.data.observation_tokenizer_panel import build_panel, PanelLimits
    begin = time.perf_counter()
    p = build_panel(ROOT/cfg['data']['g0'], limits=PanelLimits(**limits), seed=cfg['data']['panel_seed'])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(p.read, range(len(p.rows))))
    frame = []; groups = {}; audit = []
    for mod in ('eeg', 'hb'):
        groups[mod] = {}
        for split in ('train', 'validation'):
            idx = [i for i, r in enumerate(p.rows) if r.modality == mod and r.split == split]
            windows = []; masks = []; info = []; vxs = []
            for i in idx:
                s, r = samples[i], p.rows[i]
                x = np.asarray(s['x'], dtype=np.float32).transpose(0, 2, 1)
                if x.shape != (15, *cfg['model']['shapes'][mod]) or not np.isfinite(x).all():
                    raise ValueError('wrong shape or nonfinite G0 X encoder input')
                good = np.asarray(s['patch_scorable'], dtype=bool)
                if good.shape != (15,):
                    raise ValueError('expected fifteen explicit patch masks')
                available = bool(s['available'])
                meta = {k: getattr(r, k) for k in KEYS}
                row = asdict(r)
                audit.append(dict(**row, available=available, complete_scored_patches=int(good.sum()),
                                  included=available and bool(good.any()), valid_bits=''.join(str(int(v)) for v in good)))
                # Historical encoders see the entire 30 s. Exclude the same incomplete input from every arm.
                if not available or not good.any():
                    continue
                row_i = len(windows); windows.append(x); masks.append(good); vxs.append(r.vx)
                for j in np.flatnonzero(good):
                    info.append(dict(**meta, patch=int(j), window_idx=row_i))
            windows = np.stack(windows); mask = np.stack(masks); info = pd.DataFrame(info)
            if not len(info) or len(info) != int(mask.sum()) or info.duplicated(KEYS+['patch']).any():
                raise ValueError('no or ambiguous scoring support')
            variance = np.asarray(vxs, dtype=float)[info.window_idx]
            groups[mod][split] = dict(windows=windows, mask=mask, x=windows[mask], frame=info, vx=variance)
            frame.append(dict(modality=mod, split=split, windows_sites=len(windows),
                              available_patches=int(mask.sum()), excluded_patches=int((~mask).sum()),
                              subjects=len(info[KEYS[:2]].drop_duplicates()), records=len(info[KEYS[:3]].drop_duplicates())))
    pd.DataFrame(audit).to_csv(destination/'panel.csv', index=False)
    elapsed = time.perf_counter()-begin
    write_json(destination/'panel.json', dict(counts=p.counts, selected=frame, seconds=elapsed,
                                              window_sites_per_s=len(samples)/elapsed, io_workers=workers))
    return groups


def train_indices(t, cfg, smoke=False):
    n = min(1024,len(t['frame'])*2) if smoke else cfg['model']['training_draws']
    weights = hierarchy_weights(t['frame'])
    rng = np.random.default_rng(cfg['model']['training_draw_seed'])
    return rng.choice(len(weights), size=n, replace=True, p=weights)


def make_model(mod, cfg, rate, xtrain, *, seed=17, routing='pattern'):
    from src.tokenizers.factorized_observation import FactorizedObservationTokenizer as Codec
    return Codec(mod, n_patterns=cfg['model']['k'], detail_count=rate[0], detail_bits=rate[1],
                 seed=seed, routing=routing).fit(xtrain, partition='train')


def operating_point(table, mod, cfg):
    t = table[(table.level=='macro') & (table.method=='full')].sort_values('bits_per_patch')
    s = cfg['selection']; ok = (t.x_nmse <= s['max_macro_x_nmse']) & (t.centered_cosine >= s['min_macro_centered_cosine'])
    if mod == 'eeg':
        ok &= t.band_30_45_db.abs() <= s['eeg_max_abs_macro_30_45_db']
    row = t[ok].iloc[0] if ok.any() else t.iloc[-1]
    return dict(bits_per_patch=int(row.bits_per_patch), met_screen=bool(ok.any()), x_nmse=float(row.x_nmse),
                centered_cosine=float(row.centered_cosine))


def run_modality(mod, cfg, group, out, workers):
    from src.analysis.observation_legacy_control import load_legacy_observation_control
    begin = time.perf_counter(); t, v = group['train'], group['validation']; k = cfg['model']['k']
    train = np.ascontiguousarray(t['x'][train_indices(t, cfg)]); z = v['x']; f = v['frame']
    def score(pred, name, bits):
        return reconstruction(f, z, pred, v['vx'], mod).assign(method=name, bits_per_patch=bits, modality=mod, split='validation')
    def evaluate(rate):
        c = make_model(mod, cfg, rate, train)
        blind = make_model(mod, cfg, rate, train, routing='waveform')
        rate_info = {m: c.rate(m) for m in ['full', 'flat', 'prototype', 'flat_pattern']}
        n = int(rate_info['full']['bits_per_item'])
        if n != int(rate_info['flat']['bits_per_item']) or n != 4+rate[0]*rate[1]:
            raise ValueError('unequal rate control')
        ids = c.encode(z, mode='full'); pred = c.decode(ids, mode='full')
        tables = [score(pred, 'full', n), score(c.decode(c.encode(z, mode='flat'), mode='flat'), 'flat', n),
                  score(blind.decode(blind.encode(z, mode='full'), mode='full'), 'blind_full', n)]
        wrong = dict(ids, pattern=roll_patterns(f, ids['pattern']))
        tables.append(score(c.decode(wrong, mode='full'), 'shuffled_pattern', n))
        return n, c, blind, tables, rate_info
    # Independent rates share only read-only train/validation inputs and have separate model objects.
    rates = cfg['model']['rates'][mod]
    # Fitting touches process-global BLAS/OpenMP controllers in sklearn. Keep numerical
    # fits in separate worker processes; threads are used only for bounded mmap I/O.
    with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
        runs = joblib.Parallel(n_jobs=min(workers, len(rates)), pre_dispatch=workers)(
            joblib.delayed(evaluate)(rate) for rate in rates)
    n0, first, bw, _, _ = runs[0]
    a = first.encode(z, mode='prototype')['pattern']; b = bw.encode(z, mode='prototype')['pattern']
    tables = [tb for r in runs for tb in r[3]]
    for n, c, _, _, _ in runs:
        if not np.array_equal(a, c.encode(z, mode='prototype')['pattern']):
            raise ValueError('nominally common semantic channel changed with detail budget')
    for model, mode, label in [(first, 'prototype', 'prototype'), (bw, 'prototype', 'wave_kmeans'),
                               (first, 'flat_pattern', 'flat_4bit')]:
        tables.append(score(model.decode(model.encode(z, mode=mode), mode=mode), label, 4))
    # Fixed first included site/window per dataset, independent of error, activity and ID.
    examples = f.drop_duplicates('dataset'); indices = examples.window_idx.to_numpy(); saved_examples = {}
    for continuous in [False, True]:
        legacy = load_legacy_observation_control(ROOT, mod, continuous=continuous)
        p = legacy.reconstruct(v['windows']); label = 'legacy_continuous' if continuous else 'legacy_hard'
        saved_examples[label] = p[indices]; tables.append(score(p[v['mask']], label, np.nan if continuous else 14))
    table = pd.concat(tables, ignore_index=True); op = operating_point(table, mod, cfg)
    selected = next(r for r in runs if r[0] == op['bits_per_patch']); _, model, _, _, _ = selected
    example_windows = v['windows'][indices]; shape = example_windows.shape
    example_x = example_windows.reshape(-1, *shape[2:]); payload = model.encode(example_x, mode='full')
    np.savez_compressed(out/f'{mod}_first_per_dataset.npz', x=example_windows, score_mask=v['mask'][indices],
        full=model.decode(payload, mode='full').reshape(shape), pattern=payload['pattern'].reshape(-1,15),
        flat=model.decode(model.encode(example_x, mode='flat'), mode='flat').reshape(shape), **saved_examples)
    examples[KEYS].to_csv(out/f'{mod}_first_per_dataset.csv', index=False)
    model_name = out/f'{mod}_codec.joblib'; joblib.dump(model, model_name, compress=3)
    joblib.dump(selected[2], out/f'{mod}_blind_control.joblib', compress=3)
    restored = joblib.load(model_name)
    for mode in ['prototype', 'full', 'flat']:
        code = model.encode(z[:16], mode=mode); recalled = restored.encode(z[:16], mode=mode)
        if any(not np.array_equal(code[key], recalled[key]) for key in code) or not np.array_equal(
                restored.decode(recalled, mode=mode), model.decode(code, mode=mode)):
            raise ValueError('saved codec does not reproduce its outputs')
    tid = {name: c.encode(t['x'], mode='prototype')['pattern'] for name, c in [('pattern', first), ('wave_kmeans', bw)]}
    probes = probe_attributes(t['frame'], f, t['x'], z, tid, {'pattern':a, 'wave_kmeans':b}, mod, k)
    atlases = [pattern_atlas(s['frame'], ids, model.pattern_features(s['x']), model.pattern_feature_names, k, mod, split)
               for s, ids, split in [(t, tid['pattern'], 'train'), (v, a, 'validation')]]
    chosen_rate = next(rate for rate in rates if 4+rate[0]*rate[1] == op['bits_per_patch'])
    ids_by_seed = {17:a}; seed_scores = {17:op['x_nmse']}
    for seed in cfg['model']['seeds'][1:]:
        c = make_model(mod, cfg, chosen_rate, train, seed=seed); code = c.encode(z, mode='full')
        ids_by_seed[seed] = code['pattern']; rs = score(c.decode(code, mode='full'), 'full', op['bits_per_patch'])
        seed_scores[seed] = float(rs[rs.level=='macro'].iloc[0].x_nmse)
    stability = {}
    for s1, y1 in ids_by_seed.items():
        for s2, y2 in ids_by_seed.items():
            if s1 < s2:
                stability[f'{s1}_{s2}'] = {ds:float(adjusted_rand_score(y1[rows], y2[rows]))
                                           for ds, rows in f.groupby('dataset').indices.items()}
    contrasts = [paired_contrast(table, 'x_nmse', ref, cand) for ref, cand in [
        (('legacy_hard', 14), ('full', 14)), (('legacy_hard', 14), ('full', op['bits_per_patch'])),
        (('flat', op['bits_per_patch']), ('full', op['bits_per_patch'])),
        (('blind_full', op['bits_per_patch']), ('full', op['bits_per_patch']))]]
    table.to_csv(out/f'{mod}_reconstruction.csv', index=False)
    probes.to_csv(out/f'{mod}_attributes.csv', index=False)
    pd.concat(atlases, ignore_index=True).to_csv(out/f'{mod}_atlas.csv', index=False)
    stats = dict(status='completed', modality=mod, operating_point=op, model=model_name.name,
                 contrasts=contrasts, per_dataset_ARI=stability, seed_X_NMSE=seed_scores,
                 rates={r[0]: r[4] for r in runs}, train_draws=len(train),
                 val_patches=len(f), elapsed_seconds=time.perf_counter()-begin)
    write_json(out/f'{mod}_done.json', stats)
    return stats


def smoke_benchmark(cfg, g):
    # Timed end-to-end fits/encodes/decodes on the small, identity-fixed smoke panel, not accuracy selection.
    tasks = [(m, r, j) for j in range(2) for m in ['eeg', 'hb'] for r in cfg['model']['rates'][m][:2]]
    def work(args):
        mod, rate, j = args; tr = g[mod]['train']; x = tr['x'][train_indices(tr, cfg, smoke=True)]
        c = make_model(mod, cfg, rate, x, seed=17+j); z = g[mod]['validation']['x']
        return c.decode(c.encode(z, mode='full'), mode='full').shape
    times = {}
    for n in cfg['resources']['worker_candidates']:
        print(f'smoke: process concurrency benchmark n={n}', flush=True)
        with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
            with joblib.Parallel(n_jobs=n, pre_dispatch=n) as pool:
                pool(joblib.delayed(work)(a) for a in tasks[:n])  # exclude worker imports from steady-state timing
                start = time.perf_counter(); results = pool(joblib.delayed(work)(a) for a in tasks); dt = time.perf_counter()-start
        times[n] = dict(seconds=dt, fits_per_second=len(results)/dt)
        print(f'smoke: {len(results)/dt:.2f} fits/s at n={n}', flush=True)
    winner = max(times, key=lambda n: times[n]['fits_per_second'])
    return dict(status='passed', times=times, workers=winner, tasks_per_arm=len(tasks),
                reason='best measured throughput over bounded independent fits; BLAS/OpenMP=1; no validation accuracy used')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, default=CONFIG); cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--stage', required=True, choices=['synthetic', 'smoke', 'screen']); args = cli.parse_args()
    cfg = config(args.config); out = args.output.resolve()
    if out.parent != RUNS.resolve() or out == RUNS.resolve():
        raise ValueError('fresh direct child of the versioned run namespace required')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    import torch
    torch.set_num_threads(1)
    with (out/'controller.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'summary.json').exists():
            raise ValueError('completed run is immutable; select a fresh output')
        stage = args.stage
        if stage=='synthetic':
            if (out/'synthetic.json').exists():
                raise ValueError('synthetic stage already recorded')
            write_json(out/'resolved_config.json', cfg); write_json(out/'synthetic.json', synthetic(cfg)); return
        if read_json(out/'resolved_config.json') != cfg or read_json(out/'synthetic.json') != dict(
                status='passed', checks={m:'integer_shapes_rates_and_finite_outputs_passed' for m in ['eeg','hb']},
                read_measured_arrays=False, sources=source_identity()):
            raise ValueError('software or contract changed since the synthetic gate; use a fresh output')
        def progress(state, **extra):
            now = datetime.now(timezone.utc).isoformat(); obj = dict(status=state, stage=stage, time=now, **extra)
            write_json(out/'controller.json', obj); print(json.dumps(obj, default=serial, ensure_ascii=False), flush=True)
        progress('running', step='panel')
        try:
            if stage == 'smoke':
                dst=out/'smoke'; dst.mkdir(exist_ok=True); g=read_panel(cfg, cfg['data']['smoke'], dst, 2)
                write_json(out/'benchmark.json', smoke_benchmark(cfg, g)); progress('smoke_passed'); return
            bench = read_json(out/'benchmark.json')
            if bench['status'] != 'passed' or bench['workers'] not in cfg['resources']['worker_candidates']:
                raise ValueError('passed resource smoke required')
            g=read_panel(cfg, cfg['data']['panel'], out, workers=4); stats={}
            # Rate fits are parallel within each modality. Legacy inference/probes do not compete across modalities.
            for mod in ('eeg', 'hb'):
                progress('running', step=mod, workers=bench['workers'])
                if not (out/f'{mod}_done.json').is_file():
                    run_modality(mod, cfg, g[mod], out, bench['workers'])
                stats[mod] = read_json(out/f'{mod}_done.json')
            for name in ['reconstruction', 'attributes', 'atlas']:
                pd.concat([pd.read_csv(out/f'{m}_{name}.csv', dtype={'subject': str}) for m in stats],
                          ignore_index=True).to_csv(out/f'{name}.csv', index=False)
            progress('running', step='task_probe')
            from src.analysis.observation_task_probe import run_task_probe
            if not (out/'task_probe.json').is_file():
                task = run_task_probe(cache_root=ROOT/TASK_CACHE, g0_root=ROOT/G0,
                                      eeg_codec=joblib.load(out/'eeg_codec.joblib'), hb_codec=joblib.load(out/'hb_codec.joblib'),
                                      waveform_codecs={m: joblib.load(out/f'{m}_blind_control.joblib') for m in stats})
                write_json(out/'task_probe.json', task)
            write_json(out/'summary.json', dict(status='completed', experiment_id=cfg['experiment_id'],
                endpoints='exploratory held-out public development; X only; noncausal',
                sources=source_identity(), panel=read_json(out/'panel.json')['selected'], modalities=stats,
                attributes='attributes.csv', task='task_probe.json',
                peak_controller_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            progress('completed')
        except Exception:
            stamp = datetime.now().strftime('%Y%m%dT%H%M%S'); file=out/f'failure_{stage}_{stamp}.txt'
            file.write_text(traceback.format_exc()); progress('failed', error=file.name); raise


if __name__ == '__main__':
    main()
