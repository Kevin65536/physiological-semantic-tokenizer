"""Replay the retained single-VQ panel and audit frozen local attribute readouts."""
import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

import run_single_vq_capacity as capacity
from audit_joint_tokenizer_patterns import pattern_attributes
from run_dual_component_tokenizer import owner_lock, read_json, write_json
from src.data.dual_component_tokenizer import base_tokenizer_batch

CONTRACT = capacity.CONTRACT.with_name('single_vq_readout_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = capacity.ROOT / PLAN['run']
SITES = {}
PEOPLE = {}
CHUNK = 1024


def attributes(wave, mod):
    flat = wave.reshape(-1, wave.shape[-1])
    features, slices = pattern_attributes(flat, mod)
    parts = []
    fields = {}
    offset = 0
    for name in PLAN['attributes'][mod]:
        value = (np.sqrt(np.mean(flat ** 2, axis=1, keepdims=True)) if name == 'patch_rms'
                 else features[:, slice(*slices[name])])
        parts.append(value)
        fields[name] = (offset, offset + value.shape[1])
        offset += value.shape[1]
    return np.concatenate(parts, axis=1).astype(np.float32), fields


@lru_cache(maxsize=5)
def network(mod, size, seed):
    return capacity.load_scoring_model('candidate', mod, PLAN['models'][mod]['dimension'], size, seed)


@lru_cache(maxsize=8)
def retained(mod, ds, role, seed):
    spec = PLAN['models'][mod]
    name = capacity.candidate(mod, spec['dimension'], spec['size'], seed)
    directory = capacity.OUT / 'evaluation' / name
    if read_json(directory / 'completion.json')['status'] != 'completed':
        raise ValueError('incomplete parent evaluation')
    with np.load(directory / f'{mod}_{ds}_{role}.npz') as z:
        np.testing.assert_array_equal(z['examples'], capacity.PANEL['examples'][mod, role, ds])
        return {b: z[b] for b in 'SO'}


def panel():
    capacity.prepare_panel(recorded_root=PLAN['recorded_root'])
    rows = capacity.PANEL['rows']
    for mod in PLAN['models']:
        labels = sorted({(r['dataset'], p) for r in rows if r['modality'] == mod and r['split'] == 'train'
                         for p in r['positions']})
        SITES[mod] = {key: n for n, key in enumerate(labels)}
    PEOPLE.update({key: n for n, key in enumerate(sorted({(r['dataset'], r['subject']) for r in rows}))})


def shards():
    for mod in PLAN['models']:
        for ds in capacity.DATASETS:
            for role in ('train', 'validation'):
                n = len(capacity.PANEL['examples'][mod, role, ds])
                for start in range(0, n, CHUNK):
                    yield mod, ds, role, start, min(start + CHUNK, n)


def destination(task):
    mod, ds, role, start, end = task
    return OUT / 'features' / f'{mod}_{ds}_{role}_{start}_{end}.npz'


@torch.inference_mode()
def extract(task, pilot=False):
    mod, ds, role, start, end = task
    path = destination(task)
    if not pilot and path.exists():
        return
    rows = capacity.PANEL['rows']
    spec = PLAN['models'][mod]
    ex = capacity.PANEL['examples'][mod, role, ds][start:end]
    cache = OrderedDict()
    kept = defaultdict(list)
    began = time.monotonic()
    for j in range(0, len(ex), 64):
        take = ex[j:j+64]
        batch = base_tokenizer_batch(rows, take, capacity.PANEL['scales'], array_cache=cache, cache_limit=64)
        x = torch.from_numpy(batch['x']).float()
        valid = batch['mask'].all(-1)
        location = [SITES[mod].get((ds, rows[r]['positions'][p]), len(SITES[mod])) for r, _, p in take]
        bg = 4 * np.asarray(location)[:, None] + np.digitize(np.sqrt((batch['x']**2).mean(-1)), [.5, 1, 2])
        kept['bg'].append(bg[valid])
        kept['valid'].append(valid)
        for b in 'SO':
            target, _ = attributes(batch[b.lower()], mod)
            kept[b+'_target'].append(target[valid.ravel()])
        continuous = network(mod, 0, 17)(x)
        for b in 'SO':
            kept[b+'_continuous_z'].append(continuous['latent'][b].numpy()[valid])
            if role == 'validation':
                kept[b+'_continuous_wave'].append(attributes(continuous[b].numpy(), mod)[0][valid.ravel()])
        for seed in spec['seeds']:
            hard = network(mod, spec['size'], seed)
            z = hard.encode_continuous(x)
            ids = {b: torch.from_numpy(v[start+j:start+j+len(take)].astype('int64'))
                   for b, v in retained(mod, ds, role, seed).items()}
            if pilot:
                replay = hard(x)
                for b in 'SO':
                    torch.testing.assert_close(replay['ids'][b], ids[b])
            decoded = hard.decode_ids(ids) if role == 'validation' else {}
            for b in 'SO':
                key = f'{b}_{seed}_'
                kept[key+'id'].append(ids[b].numpy()[valid])
                kept[key+'z'].append(z[b].numpy()[valid])
                if decoded:
                    kept[key+'wave'].append(attributes(decoded[b].numpy(), mod)[0][valid.ravel()])
    data = {k: np.concatenate(v) for k, v in kept.items()}
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError(f'nonfinite complete-patch result in {task}')
    if not pilot:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez(temp, examples=ex, **data)
        temp.replace(path)
    return dict(modality=mod, dataset=ds, split=role, examples=len(ex), patches=int(data['valid'].sum()),
                seconds=time.monotonic()-began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pilot', 'extract', 'run', 'summarize'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with owner_lock(OUT, blocking=args.action == 'run'):
        if (OUT/'resolved_config.json').exists() and read_json(OUT/'resolved_config.json') != PLAN:
            raise ValueError('use a fresh output version for contract changes')
        write_json(OUT/'resolved_config.json', PLAN)
        if args.action == 'summarize':
            from score_single_vq_readout import summarize_scores
            summarize_scores()
            return
        panel()
        if args.action == 'pilot':
            tasks = [(m, capacity.DATASETS[0], r, 0, 64) for m in PLAN['models'] for r in ('train','validation')]
            for i, task in enumerate(tasks):
                result = extract(task, pilot=True)
                write_json(OUT/f'pilot_{i}.json', result)
                print(result, flush=True)
            return
        tasks = list(shards())
        pending = [t for t in tasks if not destination(t).exists()]
        state = dict(status='running', stage='extraction', total=len(tasks), done=len(tasks)-len(pending))
        write_json(OUT/'progress.json', state)
        try:
            ctx = multiprocessing.get_context('fork')
            with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
                for f in as_completed([pool.submit(extract, t) for t in pending]):
                    result = f.result()
                    state['done'] += 1
                    write_json(OUT/'progress.json', dict(state, last=result))
            if args.action == 'extract':
                write_json(OUT/'progress.json', dict(state, stage='features_ready'))
                return
            from score_single_vq_readout import score_all
            state.update(stage='readout')
            write_json(OUT/'progress.json', state)
            score_all(tasks, args.workers)
            write_json(OUT/'progress.json', dict(state, status='completed'))
        except BaseException as error:
            write_json(OUT/'progress.json', dict(state, status='failed', error=repr(error)))
            raise


if __name__ == '__main__':
    # Scoring imports the same populated panel and constants when run as a CLI.
    import sys
    sys.modules['run_single_vq_readout'] = sys.modules[__name__]
    main()
