#!/usr/bin/env python3
"""Train paired continuous and integer EEG models with a full-sample decoder."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import (
    _base_training_batch, identity, owner_lock, write_json,
)
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from experiments.scripts.run_joint_waveform_dictionary import preserve_record
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch, base_tokenizer_sample, base_tokenizer_sampling_index,
)
from src.tokenizers.supervised_waveform_dictionary import load_model as load_source
from src.tokenizers.fullband_eeg_tokenizer import FullbandEEGTokenizer, fullband_loss


def save(path, payload):
    temporary = path.with_suffix('.pending.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate(cfg):
    model = cfg['model']
    if (cfg['schema'] != 'dual_component_joint_neural_eeg_v1'
            or model['context_seconds'] != 30 or model['patch_seconds'] != 2
            or model['patches'] != 15 or model['input_dim'] != 400
            or model['unencoded_decoder_inputs'] or model['codebook_size'] not in (1024, 8192)
            or model['latent_dim'] != 128):
        raise ValueError('reviewed 30-second single-level fullband EEG contract required')
    if cfg['training']['stopping'] != 'fixed_steps_no_validation_extension':
        raise ValueError('fixed train-only continuation required')


def provenance(cfg, manifest, scales):
    source = ROOT / cfg['source']['checkpoint']
    source_model = load_source(source)
    if (source_model.metadata['modality'] != 'eeg'
            or source_model.metadata['input_identity'] != identity(manifest)):
        raise ValueError('source training G0 identity mismatch before arrays')
    signature = dict(config=identity(cfg), input=identity(manifest), scales=identity(scales),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if 'shared_initial_network' in cfg['source']:
        shared = ROOT / cfg['source']['shared_initial_network']
        signature['shared_initial_sha256'] = hashlib.sha256(shared.read_bytes()).hexdigest()
    return source_model, signature


def initialize(cfg, out, rows, scales, source_model, signature):
    target = out / 'initial_model.pt'
    if target.exists():
        saved = torch.load(target, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('initial checkpoint identity changed')
        return saved
    if cfg['source'].get('reuse_full_initial_state'):
        reference = torch.load(ROOT/cfg['source']['shared_initial_network'], map_location='cpu', weights_only=False)
        model_setting = {k: v for k, v in cfg['model'].items() if k != 'commitment_cost'}
        reference_setting = {k: v for k, v in reference['model_config'].items() if k != 'commitment_cost'}
        if (reference['signature']['input'] != signature['input']
                or reference['signature']['scales'] != signature['scales'] or model_setting != reference_setting):
            raise ValueError('full initialization reuse may change only commitment strength')
        saved = dict(model_config=cfg['model'], model_state=reference['model_state'],
            metadata=dict(reference['metadata'], signature=signature), signature=signature, step=0)
        save(target, saved)
        preserve_record(out/'training_initialization.json', dict(source=cfg['source'], signature=signature,
            full_network_codebook_and_EMA_state='exact_copy', new_initialization_signal_reads=0,
            changed_hyperparameter='commitment_cost'))
        return saved
    seed = cfg['training']['seed']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    examples = base_tokenizer_sample(rng, base_tokenizer_sampling_index(rows, 'eeg'),
                                    cfg['training']['initialization_contexts'])
    if any(rows[i]['split'] != 'train' for i, _, _ in examples):
        raise ValueError('initialization train-only scope before signal reads')
    cache, parts = OrderedDict(), []
    for offset in range(0, len(examples), 128):
        parts.append(base_tokenizer_batch(rows, examples[offset:offset+128], scales, array_cache=cache,
                                         cache_limit=cfg['resources']['mmap_parent_cache']))
    batch = {k: np.concatenate([v[k] for v in parts]) for k in ('x', 's', 'o', 'mask')}
    del parts
    predicted = source_model.predict_continuous(batch['x'])
    valid = batch['mask'].all(-1)
    values = dict(transfer=np.stack([source_model.transfer[b][:, 0, 0] for b in ('S', 'O')]))
    band_floor, rms_floor = [], []
    hann = np.hanning(400)
    freq = np.fft.rfftfreq(400, d=1/200)
    for b in ('S', 'O'):
        basis, center = source_model.bases[b], source_model.means[b]
        projected = (predicted[b][valid]-center) @ basis.T
        std = np.std(projected, axis=0)
        scale = np.maximum(std, np.median(std)*.1)**cfg['model']['latent_scale_exponent']
        values.update({b+'_basis': basis, b+'_mean': center, b+'_scale': scale})
        y = batch[b.lower()][valid]
        power = np.abs(np.fft.rfft((y-y.mean(-1, keepdims=True))*hann))**2
        floors = []
        for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45.001)):
            band = (freq >= lo) & (freq < hi)
            floors.append(max(float(np.median(power[:, band].sum(-1))), 1e-6))
        band_floor.append(floors)
        rms_floor.append(max(float(np.median(np.sqrt(np.mean(y*y, axis=-1)))), 1e-4))
    values.update(band_floor=np.asarray(band_floor), rms_floor=np.asarray(rms_floor))
    model = FullbandEEGTokenizer(cfg['model'], values)
    if 'shared_initial_network' in cfg['source']:
        reference = torch.load(ROOT/cfg['source']['shared_initial_network'], map_location='cpu', weights_only=False)
        model_setting = {k: v for k, v in cfg['model'].items() if k != 'codebook_size'}
        reference_setting = {k: v for k, v in reference['model_config'].items() if k != 'codebook_size'}
        if (reference['signature']['input'] != signature['input']
                or reference['signature']['scales'] != signature['scales'] or model_setting != reference_setting):
            raise ValueError('shared network changes more than the declared codebook size')
        retained = {k: v for k, v in reference['model_state'].items() if '.quantizer.' not in k}
        loaded = model.load_state_dict(retained, strict=False)
        if loaded.unexpected_keys or any('.quantizer.' not in key for key in loaded.missing_keys):
            raise ValueError('shared initial nonquantizer weights mismatch')
    model.eval()
    latents = {b: [] for b in ('S', 'O')}
    with torch.inference_mode():
        for offset in range(0, len(examples), 128):
            encoded = model.encode_continuous(torch.as_tensor(batch['x'][offset:offset+128], dtype=torch.float32))
            for b in ('S', 'O'):
                latents[b].append(encoded[b].numpy()[valid[offset:offset+128]])
    fitted = {}
    for b in ('S', 'O'):
        z = np.concatenate(latents[b])
        start = time.monotonic()
        fitting = MiniBatchKMeans(n_clusters=cfg['model']['codebook_size'], n_init=1, random_state=seed,
            batch_size=cfg['training']['initialization_kmeans_batch'],
            max_iter=cfg['training']['initialization_kmeans_max_iter'], reassignment_ratio=0.)
        labels = fitting.fit_predict(z)
        count = np.bincount(labels, minlength=cfg['model']['codebook_size']).astype(np.float32)
        quantizer = model.branches[b]['quantizer']
        with torch.no_grad():
            quantizer.codebook.copy_(torch.as_tensor(fitting.cluster_centers_))
            quantizer.ema_count.copy_(torch.as_tensor(np.maximum(count, 1.)))
            quantizer.ema_sum.copy_(quantizer.codebook*quantizer.ema_count[:, None])
            quantizer.initialized.fill_(True)
        fitted[b] = dict(patches=len(z), occupied_codes=int((count>0).sum()), inertia=fitting.inertia_,
                         kmeans_steps=int(fitting.n_iter_), seconds=time.monotonic()-start)
    metadata = dict(schema='fullband_eeg_tokenizer_v1', modality='eeg', codebook_size=cfg['model']['codebook_size'],
        cfg_identity=signature['input_cfg'], scales_identity=signature['scales'], input_identity=signature['input'],
        signature=signature, input='X_only', context_seconds=30, patches=15,
        reconstruction_coordinate='G0_fixed_train_position_input_SD')
    initial = dict(model_config=cfg['model'], model_state=model.state_dict(), metadata=metadata,
                   signature=signature, step=0)
    save(target, initial)
    preserve_record(out/'training_initialization.json', dict(examples=examples, planned_contexts=len(examples),
        complete_patches=int(valid.sum()), total_patches=int(valid.size), codebooks=fitted,
        fitting_split='train', validation_arrays_read=0, source=cfg['source'], signature=signature))
    return initial


def training_model(initial):
    model = FullbandEEGTokenizer(initial['model_config'])
    model.load_state_dict(initial['model_state'])
    return model


def monitor(model, batch, cfg, hard):
    model.eval()
    with torch.inference_mode():
        prediction = model(batch['x'], quantize=hard)
        _, terms = fullband_loss(model, prediction, batch, cfg['training']['loss_weights'])
        if hard:
            replay = model.decode_ids(prediction['ids'])
            terms['integer_replay_max_abs'] = max(float(abs(prediction[b]-replay[b]).max()) for b in ('S', 'O', 'sum'))
            if terms['integer_replay_max_abs'] != 0:
                raise AssertionError('frozen integer lookup must reproduce forward exactly')
    return terms


def run(args, cfg):
    validate(cfg)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with owner_lock(out, 'initialization', blocking=True):
        preserve_record(out/'resolved_config.json', cfg)
        manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
        rows = manifest['rows']
        source, signature = provenance(cfg, manifest, scales)
        signature['input_cfg'] = manifest['cfg_identity']
        initial = initialize(cfg, out, rows, scales, source, signature)
    groups, cache = base_tokenizer_sampling_index(rows, 'eeg'), OrderedDict()
    fixed = base_tokenizer_sample(np.random.default_rng(8043), groups, cfg['training']['monitor_examples'])
    fixed_batch = _base_training_batch(rows, fixed, scales, 'cpu', cache, cfg)
    if args.mode == 'prepare':
        model = training_model(initial)
        write_json(out/'initial_diagnostics.json', dict(continuous=monitor(model, fixed_batch, cfg, False),
            hard=monitor(model, fixed_batch, cfg, True), source='same_64_train_draws_not_G0_hierarchical_score'))
        return
    if args.mode == 'benchmark':
        timings = []
        for threads in cfg['resources']['benchmark_threads']:
            torch.set_num_threads(threads)
            model = training_model(initial).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            started = time.monotonic()
            for _ in range(cfg['resources']['benchmark_steps']):
                prediction = model(fixed_batch['x'][:32], quantize=True, valid_mask=fixed_batch['mask'][:32].any(-1))
                loss, _ = fullband_loss(model, prediction, {k: v[:32] for k, v in fixed_batch.items()}, cfg['training']['loss_weights'])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            duration = time.monotonic()-started
            timings.append(dict(threads=threads, steps_per_second=cfg['resources']['benchmark_steps']/duration))
        write_json(out/'benchmark.json', dict(timings=timings, batch_size=32, type='hard_full_training_step'))
        return
    if args.candidate not in ('continuous', 'vq'+str(cfg['model']['codebook_size'])):
        raise ValueError('candidate and configured atomic vocabulary differ')
    hard = args.candidate != 'continuous'
    job = out/args.candidate
    job.mkdir(exist_ok=True)
    with owner_lock(job):
        model = training_model(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                     weight_decay=cfg['training']['weight_decay'])
        rng = np.random.default_rng(cfg['training']['seed'])
        torch.manual_seed(cfg['training']['seed'])
        checkpoint, start, history = job/'checkpoint.pt', 0, []
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['signature'] != signature or saved['metadata']['candidate'] != args.candidate:
                raise ValueError('continuation checkpoint identity changed')
            model.load_state_dict(saved['model_state'])
            optimizer.load_state_dict(saved['optimizer_state'])
            rng.bit_generator.state = saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng'])
            start, history = saved['step'], saved['history']
        if (job/'summary.json').exists():
            if start != cfg['training']['steps']:
                raise ValueError('completed summary with incomplete optimizer checkpoint')
            return
        initial_monitor = monitor(model, fixed_batch, cfg, hard)
        begin = time.monotonic()
        metadata = dict(initial['metadata'], candidate=args.candidate)
        for step in range(start+1, cfg['training']['steps']+1):
            if time.monotonic()-begin > cfg['resources']['wall_budget_seconds']:
                raise TimeoutError('fixed wall budget exhausted; last checkpoint retained')
            examples = base_tokenizer_sample(rng, groups, cfg['training']['batch_size'])
            batch = _base_training_batch(rows, examples, scales, 'cpu', cache, cfg)
            model.train()
            prediction = model(batch['x'], quantize=hard, valid_mask=batch['mask'].any(-1))
            loss, terms = fullband_loss(model, prediction, batch, cfg['training']['loss_weights'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite fullband training loss')
            fraction = step/cfg['training'].get('learning_rate_schedule_steps', cfg['training']['steps'])
            rate = cfg['training']['final_learning_rate'] + .5*(cfg['training']['learning_rate']-cfg['training']['final_learning_rate'])*(1+math.cos(math.pi*fraction))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
            optimizer.step()
            if step == 1 or step % 64 == 0:
                history.append(dict(step=step, elapsed_seconds=time.monotonic()-begin, **terms))
                write_json(job/'progress.json', dict(status='training', step=step, total=cfg['training']['steps'],
                    elapsed_seconds=time.monotonic()-begin, last=terms))
            if step in cfg['training']['milestones'] or step % 128 == 0:
                retained = dict(model_config=cfg['model'], model_state=model.state_dict(), metadata=metadata,
                    signature=signature, step=step, optimizer_state=optimizer.state_dict(),
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), history=history)
                save(checkpoint, retained)
                if step in cfg['training']['milestones']:
                    save(job/f'checkpoint_step_{step:05d}.pt', retained)
                    write_json(job/f'monitor_step_{step:05d}.json', monitor(model, fixed_batch, cfg, hard))
        save(job/'model.pt', dict(model_config=cfg['model'], model_state=model.state_dict(), metadata=metadata,
            signature=signature, step=cfg['training']['steps']))
        write_json(job/'summary.json', dict(status='completed', candidate=args.candidate,
            completed_step=cfg['training']['steps'], initial_train_monitor=initial_monitor,
            final_train_monitor=monitor(model, fixed_batch, cfg, hard),
            parameters=sum(p.numel() for p in model.parameters()), source=cfg['source'],
            elapsed_seconds=time.monotonic()-begin, validation_arrays_read=0,
            source_initialization='fixed_train_Wiener_and_target_PCA_then_train_only_EMA', signature=signature))
        write_json(job/'progress.json', dict(status='completed', step=cfg['training']['steps']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_neural_eeg_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'benchmark', 'train'], required=True)
    parser.add_argument('--candidate', choices=['continuous', 'vq1024', 'vq8192'], default='vq1024')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(cfg['resources']['torch_threads'])
    with threadpool_limits(cfg['resources']['numerical_threads']):
        run(args, cfg)


if __name__ == '__main__':
    main()
