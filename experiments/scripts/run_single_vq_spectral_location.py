"""Matched EEG-S spectral loss placement on the capacity study's ordinary development panel."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

import run_single_vq_capacity as capacity
from run_dual_component_tokenizer import (
    _base_training_model, _state_refinement_loss, now, owner_lock, read_json, train_base_tokenizer, write_json,
)
from src.analysis.dual_component_tokenizer import eeg_spectral_attributes
from src.data.dual_component_tokenizer import base_tokenizer_batch
from evaluate_joint_tokenizer import input_scales

ROOT = capacity.ROOT
CONTRACT = ROOT / 'experiments/configs/physiology_semantic_tokenizer/single_vq_spectral_location_v1.yaml'
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT / PLAN['run']
K = PLAN['interface']['size']
DIM = PLAN['interface']['dimension']
TASK = capacity.model_task('eeg', K)


def check():
    parent = capacity.PLAN
    t = PLAN['training']
    if ('null' not in PLAN or PLAN['schema'] != 'single_vq_spectral_location_v1' or ROOT / PLAN['parent'] != capacity.CONTRACT
            or PLAN['arms'] != dict(A='waveform', B='decoded_spectrum', C='single_id_spectrum')
            or K != 8192 or DIM != 64 or PLAN['interface']['width'] != parent['width']
            or (PLAN['interface']['patch_samples'], PLAN['interface']['patches']) != (400, 15)
            or [t['first_seed'], t['repeat_seed']] != parent['seeds']
            or any(t[k] != parent['training'][k] for k in ('batch_size', 'warmup_steps', 'total_steps'))
            or PLAN['scope']['development_eeg_positions'] != parent['support']['eeg_positions']
            or any(PLAN['scope'][k] != parent['scope'][k] for k in parent['scope'])
            or not 0 < PLAN['weight'] < 1):
        raise ValueError('unexpected spectral placement contract')


def training_cfg(seed):
    cfg = capacity.resolved('eeg', DIM, seed)
    cfg['experiment_id'] = PLAN['experiment_id']
    return cfg


def folder(seed, arm):
    return OUT / f'seed_{seed}' / arm


def calibrate():
    capacity.prepare_panel(recorded_root=PLAN['recorded_root'])
    panel = capacity.PANEL
    rows, scales = panel['rows'], panel['scales']
    result = {}
    cache = OrderedDict()
    for ds in capacity.DATASETS:
        values = []
        examples = panel['examples']['eeg', 'train', ds]
        for i in range(0, len(examples), 64):
            ex = examples[i:i+64]
            if any(rows[r]['split'] != 'train' for r, _, _ in ex):
                raise ValueError('training examples required for feature scales')
            batch = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
            v = torch.from_numpy(batch['s']).double()
            scale = torch.from_numpy(input_scales(rows, ex, scales))
            values.append(eeg_spectral_attributes(v, scale).numpy()[batch['mask'].all(-1)])
        array = np.concatenate(values)
        mean, sd = array.mean(0), array.std(0)
        if not np.isfinite(array).all() or not np.all(sd > 0):
            raise ValueError(f'invalid training feature calibration for {ds}')
        result[ds] = dict(mean=mean.tolist(), sd=sd.tolist(), patches=len(array), examples=len(examples))
    write_json(OUT/'calibration.json', result)


def train(seed, arm):
    cfg = training_cfg(seed)
    base = OUT / f'seed_{seed}'
    if arm == 'warmup':
        write_json(base/'resolved_training.json', cfg)
        return train_base_tokenizer(cfg, ROOT, base, dict(TASK, candidate='warmup'), 'cuda:0',
                                    recorded_root=PLAN['recorded_root'])
    objective = None if arm == 'A' else dict(location=PLAN['arms'][arm], weight=PLAN['weight'],
                                              calibration=read_json(OUT/'calibration.json'))
    return train_base_tokenizer(cfg, ROOT, base, TASK, 'cuda:0', destination=folder(seed, arm),
                                recorded_root=PLAN['recorded_root'], spectral_objective=objective)


def synthetic(arm, device, steps, batch_size):
    # Exact candidate graph and K; random inputs, dictionary and teacher, including a partial patch.
    cfg = yaml.safe_load((capacity.CONTRACT.parent/'dual_component_base_g1_v2.yaml').read_text())
    cfg['architectures']['waveform'].update(width=256, latent_dim=DIM)
    cfg['model']['geometry_each_forward'] = False
    cfg['capacity'] = dict(sizes=[K], uniform_count_fraction=.1)
    torch.manual_seed(17)
    model = _base_training_model(cfg, TASK, single_id_spectrum=arm == 'C').to(device)
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    x = torch.randn(batch_size, 15, 400, device=device)
    batch = dict(x=x, s=x*.4, o=x*.6, mask=torch.ones_like(x, dtype=torch.bool),
                 input_scale=x.new_ones(batch_size), feature_mean=x.new_zeros(batch_size, 1, 4),
                 feature_sd=x.new_ones(batch_size, 1, 4))
    batch['mask'][0, 0, -1] = False
    for key in 'xso':
        batch['v'+key] = torch.ones_like(x)
        batch['dv'+key] = x.new_ones(batch_size, 1)
    spec = dict(cfg['architectures']['waveform'])
    if arm != 'A':
        spec['spectral_objective'] = dict(location=PLAN['arms'][arm], weight=PLAN['weight'])
    opt = torch.optim.AdamW(model.parameters(), lr=.0003)
    for i in range(steps+1):
        if i == 1:
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            began = time.monotonic()
        opt.zero_grad(set_to_none=True)
        result = model(x)
        loss, _ = _state_refinement_loss(result, batch, spec)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite synthetic gradients')
        opt.step()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.monotonic()-began
    with torch.no_grad():
        model.eval()
        result = model(x)
        for b, wave in model.decode_ids(result['ids']).items():
            if b in ('S', 'O', 'sum'):
                torch.testing.assert_close(wave, result[b], rtol=0, atol=0)
        if arm == 'C':
            entry = model.branches['S']['quantizer'].get_embedding(result['ids']['S'])
            torch.testing.assert_close(model.spectral_head(entry), result['S_spectrum'], rtol=0, atol=0)
    return dict(status='passed', arm=arm, steps_per_second=steps/seconds, batch_size=batch_size,
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), peak_reserved_bytes=(
                    torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0))


def worker(action, seed, arm, gpu):
    argv = [sys.executable, '-u', str(Path(__file__).resolve()), action, '--seed', str(seed), '--arm', arm]
    with (OUT/f'{action}_{seed}_{arm}.log').open('a') as log:
        subprocess.run(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def controller():
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader'], text=True)
    admitted = {u for bus, u in (s.split(', ') for s in query.splitlines()) if bus in (
        '00000000:4B:00.0', '00000000:CA:00.0')}
    if len(visible) != 2 or set(visible) != admitted:
        raise ValueError('two permitted PCI-selected UUIDs required')
    with owner_lock(OUT), ThreadPoolExecutor(3) as pool:
        frozen = OUT/'resolved_config.json'
        if frozen.exists() and read_json(frozen) != PLAN:
            raise ValueError('contract differs from launched run')
        write_json(frozen, PLAN)
        first, repeat = PLAN['training']['first_seed'], PLAN['training']['repeat_seed']
        def phase(stage, seed, arms):
            write_json(OUT/'stage.json', dict(stage=stage, seed=seed, arms=arms, updated=now()))
        def parallel(action, seed, arms):
            futures = [pool.submit(worker, action, seed, arm, visible[i % 2]) for i, arm in enumerate(arms)]
            for f in futures:
                f.result()
        phase('synthetic_resource_pilot', first, list(PLAN['arms']))
        if not all((OUT/f'pilot_{a}.json').is_file() for a in PLAN['arms']):
            parallel('pilot', first, list(PLAN['arms']))
        phase('training_only_calibration_and_warmup', first, ['warmup'])
        calibration = None if (OUT/'calibration.json').exists() else pool.submit(worker, 'calibrate', first, 'A', '')
        worker('train', first, 'warmup', visible[0])
        if calibration is not None:
            calibration.result()
        def fit_and_score(seed, arms):
            phase('matched_training', seed, arms)
            parallel('train', seed, arms)
            phase('development_scoring', seed, arms)
            parallel('score', seed, arms)
            from score_single_vq_spectral_location import compare
            return compare(OUT, seed, arms, PLAN)
        screen = fit_and_score(first, list(PLAN['arms']))
        write_json(OUT/'selection.json', screen)
        chosen = screen['selected']
        replica = None
        if chosen:
            phase('independent_warmup', repeat, ['warmup'])
            worker('train', repeat, 'warmup', visible[0])
            replica = fit_and_score(repeat, ['A', chosen])
        write_json(OUT/'summary.json', dict(status='completed', first_seed=screen, repeat_seed=replica,
                   historical_A=capacity.summary('eeg', DIM, K, first)['selection_score'], finished=now()))
        phase('completed', first, list(PLAN['arms']))
        from score_single_vq_spectral_location import register_completion
        register_completion(OUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['controller', 'synthetic', 'pilot', 'calibrate', 'train', 'score'])
    parser.add_argument('--arm', choices=['warmup', *PLAN['arms']], default='A')
    parser.add_argument('--seed', type=int, choices=capacity.PLAN['seeds'], default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check()
    if args.action == 'controller':
        try:
            controller()
        except Exception:
            write_json(OUT/f'failure_{time.time_ns()}.json', dict(error=traceback.format_exc(), updated=now()))
            raise
    elif args.action == 'synthetic':
        print([synthetic(a, 'cpu', 1, 2) for a in PLAN['arms']])
    elif args.action == 'pilot':
        write_json(OUT/f'pilot_{args.arm}.json', synthetic(args.arm, 'cuda:0', 24, 64))
    elif args.action == 'calibrate':
        calibrate()
    elif args.action == 'train':
        train(args.seed, args.arm)
    else:
        from score_single_vq_spectral_location import score
        score(args.seed, args.arm)


if __name__ == '__main__':
    main()
