"""Complete the matched seed-17 continuous/hard × decoded EEG-S objective comparison."""

import argparse
from collections import OrderedDict, defaultdict
import os
import subprocess

import numpy as np
import torch
import yaml

import run_single_vq_spectral_location as parent
from run_dual_component_tokenizer import (
    _base_training_model,
    identity,
    owner_lock,
    read_json,
    train_base_tokenizer,
    write_json,
)
from score_single_vq_spectral_location import patch_means
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    base_training_metrics,
    eeg_spectral_attributes,
)
from src.data.dual_component_tokenizer import base_tokenizer_batch, eeg_spectrum_coordinates
from src.utils.io import save_npz_atomic

CONTRACT = parent.CONTRACT.with_name('single_vq_dynamic_factorial_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = parent.ROOT / PLAN['run']
SEED = PLAN['seed']
TASK = parent.capacity.model_task('eeg', 0)
ARMS = ('continuous_wave', 'continuous_dynamic')


def check():
    parent.check()
    expected_arms = dict(
        continuous_wave='continuous', continuous_dynamic='continuous', hard_wave='A', hard_dynamic='B'
    )
    if (
        PLAN['schema'] != 'single_vq_dynamic_factorial_v1'
        or PLAN['modality'] != 'eeg'
        or PLAN['arms'] != expected_arms
        or PLAN['scope'] != parent.PLAN['scope']
        or parent.ROOT / PLAN['parent'] != parent.CONTRACT
        or PLAN['features'] != parent.PLAN['features']
        or (SEED, PLAN['width'], PLAN['latent_dimension'], PLAN['hard_codebook'])
        != (17, 256, parent.DIM, parent.K)
        or any(v != parent.PLAN['training'][k] for k, v in PLAN['training'].items())
        or parent.PLAN['weight'] != 0.01
        or parent.PLAN['arms']['B'] != 'decoded_spectrum'
    ):
        raise ValueError('factorial requires the fixed same-waveform spectral parent')


def objective(arm):
    return (
        dict(
            location='decoded_spectrum',
            weight=parent.PLAN['weight'],
            calibration=read_json(parent.OUT / 'calibration.json'),
        )
        if arm == 'continuous_dynamic'
        else None
    )


def check_gpu():
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader'], text=True
    )
    allowed = {
        line.split(', ')[1]
        for line in result.splitlines()
        if line.split(', ')[0] in ('00000000:4B:00.0', '00000000:CA:00.0')
    }
    if gpu not in allowed:
        raise ValueError('a single permitted PCI-resolved GPU UUID is required')


def train(arm):
    cfg = parent.training_cfg(SEED)
    # Use the parent's exact warmup and config signature; only the destination is new.
    train_base_tokenizer(
        cfg,
        parent.ROOT,
        parent.OUT / f'seed_{SEED}',
        TASK,
        'cuda:0',
        destination=OUT / arm,
        recorded_root=parent.PLAN['recorded_root'],
        spectral_objective=objective(arm),
    )


@torch.inference_mode()
def score(arm):
    cfg = parent.training_cfg(SEED)
    path = OUT / arm
    model = _base_training_model(cfg, TASK).eval().to('cuda:0')
    state = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)
    s = read_json(path / 'summary.json')
    obj = objective(arm)
    if (
        s['status'] != 'completed'
        or state['step'] != 10000
        or state['signature'] != s['signature']
        or state['signature']['config'] != identity(cfg)
        or state['signature']['task'] != TASK
        or state['signature'].get('spectral_objective') != (identity(obj) if obj else None)
    ):
        raise ValueError('wrong terminal continuous model')
    model.load_state_dict(state['model_state'])
    parent.capacity.prepare_panel(recorded_root=parent.PLAN['recorded_root'])
    panel = parent.capacity.PANEL
    rows, scales = panel['rows'], panel['scales']
    calibration = read_json(parent.OUT / 'calibration.json')
    metrics = defaultdict(list)
    examples = []
    dest = path / 'evaluation'
    dest.mkdir(exist_ok=True)
    for ds in parent.capacity.DATASETS:
        ex = panel['examples']['eeg', 'validation', ds]
        file = dest / (ds + '.npz')
        if not file.exists():
            parts = defaultdict(list)
            cache = OrderedDict()
            for start in range(0, len(ex), 64):
                take = ex[start : start + 64]
                raw = base_tokenizer_batch(rows, take, scales, array_cache=cache)
                x = torch.as_tensor(raw['x'], dtype=torch.float32, device='cuda:0')
                res = model(x)
                coords = eeg_spectrum_coordinates(rows, take, scales, calibration)
                scale, mean, sd = [
                    x.new_tensor(coords[k]) for k in ('input_scale', 'feature_mean', 'feature_sd')
                ]

                def features(wave):
                    return ((eeg_spectral_attributes(wave, scale) - mean) / sd).cpu().numpy()

                valid = raw['mask'].all(-1)
                target = features(x.new_tensor(raw['s']))[valid]
                error = features(res['S'])[valid] - target
                if not np.isfinite(error).all():
                    raise ValueError('nonfinite spectral score')
                parts['target'].append(target)
                parts['valid'].append(valid)
                parts['decoded'].append(patch_means(error**2, valid))
                pred = {k: res[k].cpu().numpy() for k in ('S', 'O', 'sum')}
                m = base_training_metrics(pred, raw, 1)
                for name in ('nmse', 'correlation'):
                    parts[name].append(m[name])
            save_npz_atomic(file, examples=ex, **{k: np.concatenate(v) for k, v in parts.items()})
        with np.load(file) as data:
            np.testing.assert_array_equal(ex, data['examples'])
            for name in ('nmse', 'correlation', 'decoded'):
                metrics[name].append(data[name])
        examples.append(ex)
    result = base_training_aggregate(
        {k: np.concatenate(v) for k, v in metrics.items()}, np.concatenate(examples), rows
    )
    if (
        result['subjects'] != 16
        or result['inputs'] != 53476
        or not np.isfinite(result['metrics']['decoded']['macro']).all()
    ):
        raise ValueError('incomplete development scoring')
    write_json(dest / 'summary.json', dict(result, status='completed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'summarize'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    check()
    torch.set_num_threads(1)
    with owner_lock(OUT, 'config', blocking=True):
        path = OUT / 'resolved_config.json'
        if path.exists() and read_json(path) != PLAN:
            raise ValueError('contract changed')
        write_json(path, PLAN)
    if args.action == 'summarize':
        from score_single_vq_dynamic_factorial import summarize

        with owner_lock(OUT, 'summarize'):
            summarize()
        return
    if args.arm is None:
        parser.error('--arm is required for run')
    check_gpu()
    with owner_lock(OUT, args.arm):
        status = OUT / (args.arm + '_status.json')
        try:
            write_json(status, dict(status='running', stage='training'))
            train(args.arm)
            write_json(status, dict(status='running', stage='scoring'))
            score(args.arm)
            write_json(status, dict(status='completed'))
        except BaseException as exc:
            write_json(status, dict(status='failed', error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
