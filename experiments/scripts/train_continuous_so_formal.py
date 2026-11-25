"""Full G0 support, final-C objective and two matched formal seed budgets."""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import argparse
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_single_vq_spectral_location as formal
from experiments.scripts.check_continuous_so_targets import CFG, OUT, ORDER, admissible
from src.data.temporal_projection_so import continuous_projection
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_scales,
    base_tokenizer_batch,
)
from src.data.so_supervision_cache import write_json
from run_dual_component_tokenizer import read_json, train_base_tokenizer, owner_lock
from src.analysis.dual_component_tokenizer import eeg_spectral_attributes
from evaluate_joint_tokenizer import input_scales

CACHE = OUT / 'cache'


def load_parent():
    return base_tokenizer_training_data(
        formal.training_cfg(17), ROOT, recorded_root=formal.PLAN['recorded_root']
    )


def cache_row(item):
    i, row = item
    folder = CACHE / str(i)
    done = folder / 'ready.json'
    if done.exists():
        return read_json(done)
    folder.mkdir(parents=True, exist_ok=True)
    src = Path(row['folder'])
    for k in ['x', 'mask']:
        dst = folder / (k + '.npy')
        if not dst.exists():
            os.link(src / (k + '.npy'), dst)
        if not os.path.samefile(src / (k + '.npy'), dst):
            raise ValueError('wrong input link')
    x, old, mask = (np.load(src / (k + '.npy'), mmap_mode='r')[:, :, 0] for k in ['x', 's', 'mask'])
    valid = admissible(row, mask)
    s = continuous_projection(old, valid, ORDER)
    o = np.where(valid, x - s, 0.0)
    for k, v in [('s', s), ('o', o)]:
        part = folder / (k + '.partial.npy')
        np.save(part, v[:, :, None])
        os.replace(part, folder / (k + '.npy'))
    moments = {}
    if row['split'] == 'train':
        for k, arr in [('x', x), ('s', s), ('o', o)]:
            for diff in (False, True):
                agg = np.zeros((3, x.shape[1]))
                for g in row['inputs']['score']:
                    a = g['start_s'] * 200
                    sl = slice(a, a + 6000)
                    pp = g['positions']
                    if not pp:
                        continue
                    v = arr[sl, pp].astype(float)
                    m = valid[sl, pp]
                    np.testing.assert_array_equal(m, mask[sl, pp])
                    if diff:
                        v = np.diff(v, axis=0)
                        m = m[1:] & m[:-1]
                    v = np.where(m, v, 0.0)
                    agg[:, pp] += np.stack([m.sum(0), v.sum(0), np.square(v).sum(0)])
                moments[('d' if diff else '') + k] = {
                    field: vec[:, None].tolist() for field, vec in zip(['count', 'sum', 'second'], agg)
                }
    rec = dict(i=i, parent=row['parent'], split=row['split'], moments=moments)
    write_json(done, rec)
    return rec


def target_view(manifest, scales):
    for i, row in enumerate(manifest['rows']):
        if row['modality'] == 'eeg':
            row['folder'] = str(CACHE / str(i))
    manifest['target_version'] = CFG['experiment_id']
    updated = read_json(OUT / 'training_scales.json')
    for ds, sites in scales['positions']['eeg'].items():
        for name, ref in sites.items():
            new = updated['positions']['eeg'][ds][name]
            assert ref['scale'] == new['scale'] and ref['target_variance']['x'] == new['target_variance']['x']
    return manifest, updated


def overlay():
    manifest, scales, audit, parent = load_parent()
    manifest, scales = target_view(manifest, scales)
    return manifest, scales, audit, parent


def prepare(workers):
    assert read_json(OUT / 'target_gate.json')['advance'] and read_json(OUT / 'resolved.json') == CFG
    if (OUT / 'prepared.json').exists():
        return read_json(OUT / 'prepared.json')
    manifest, old, _, _ = load_parent()
    rows = manifest['rows']
    work = [(i, r) for i, r in enumerate(rows) if r['modality'] == 'eeg']
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, rec in enumerate(pool.map(cache_row, work, chunksize=4), 1):
            i = rec['i']
            assert rows[i]['parent'] == rec['parent']
            rows[i]['moments'] = rec['moments']
            if n % 200 == 0:
                print(n, len(work), round(time.monotonic() - start, 1), flush=True)
    spec = yaml.safe_load((ROOT / formal.training_cfg(17)['data']['contract']).read_text())['normalization']
    fit = base_tokenizer_scales([r for _, r in work], spec, input_scales=old)
    scales = copy.deepcopy(old)
    for ds, refs in fit['positions']['eeg'].items():
        for site, ref in refs.items():
            dst = scales['positions']['eeg'][ds][site]
            for k in ['target_variance', 'difference_variance']:
                for b in ['s', 'o']:
                    dst[k][b] = ref[k][b]
    write_json(OUT / 'training_scales.json', scales)
    formal.capacity.prepare_panel(recorded_root=formal.PLAN['recorded_root'])
    panel = formal.capacity.PANEL
    for i, r in work:
        r['folder'] = str(CACHE / str(i))
    calibration = {}
    cache = OrderedDict()
    for ds in formal.capacity.DATASETS:
        ex = panel['examples']['eeg', 'train', ds]
        v = []
        for a in range(0, len(ex), 64):
            batch = ex[a : a + 64]
            raw = base_tokenizer_batch(rows, batch, scales, array_cache=cache)
            values = eeg_spectral_attributes(
                torch.from_numpy(raw['s']).double(), torch.from_numpy(input_scales(rows, batch, scales))
            )
            v.append(values.numpy()[raw['mask'].all(-1)])
        v = np.concatenate(v)
        calibration[ds] = dict(
            mean=v.mean(0).tolist(), sd=v.std(0).tolist(), patches=len(v), examples=len(ex)
        )
    write_json(OUT / 'calibration.json', calibration)
    write_json(
        OUT / 'prepared.json',
        dict(
            status='completed',
            parents=len(work),
            seconds=time.monotonic() - start,
            source=formal.PLAN['parent'],
            identity='same G0 rows, admissible sampling triples, initialization, masks, X and X scales',
        ),
    )


def train(seed):
    assert read_json(OUT / 'prepared.json')['status'] == 'completed'
    cfg = formal.training_cfg(seed)
    cfg['experiment_id'] = CFG['experiment_id']
    source = dict(recorded_root=formal.PLAN['recorded_root'], target_view=target_view)
    dest = OUT / f'seed_{seed}'
    write_json(dest / 'training_config.json', cfg)
    train_base_tokenizer(cfg, ROOT, dest, dict(formal.TASK, candidate='warmup'), 'cuda:0', **source)
    objective = dict(
        location='single_id_spectrum',
        weight=formal.PLAN['weight'],
        calibration=read_json(OUT / 'calibration.json'),
    )
    train_base_tokenizer(
        cfg, ROOT, dest, formal.TASK, 'cuda:0', destination=dest / 'C', spectral_objective=objective, **source
    )
    # Match initialization sample identities, independent of target and learned centroids.
    assert read_json(dest / 'C' / 'initialization.json') == read_json(
        formal.folder(seed, 'C') / 'initialization.json'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train'])
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, choices=[17, 43], default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'prepare':
        with owner_lock(OUT, 'prepare'):
            prepare(args.workers)
    else:
        train(args.seed)
