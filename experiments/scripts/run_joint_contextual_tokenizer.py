#!/usr/bin/env python3
"""Fit larger single-level latent dictionaries on the completed public G0 train split."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '4')

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
import yaml

from run_dual_component_tokenizer import (
    SOURCE, _base_training_model, _base_training_batch, _state_refinement_loss, write_json, serial, identity,
)
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.data.dual_component_tokenizer import base_tokenizer_sample, base_tokenizer_sampling_index
from src.tokenizers.contextual_latent_dictionary import (load_model, hb_semantic_features, SemanticContextualDictionary)


def save(path, value):
    temporary = path.with_suffix('.temporary.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def configured_model(source_config, modality, size, source):
    cfg = deepcopy(source_config)
    cfg['model']['codebook_size'] = size
    model = _base_training_model(cfg, dict(modality=modality, architecture='waveform', candidate=f'vq{size}'))
    state = {key: value for key, value in source['model_state'].items() if '.quantizer.' not in key}
    loaded = model.load_state_dict(state, strict=False)
    if loaded.unexpected_keys or any('.quantizer.' not in name for name in loaded.missing_keys):
        raise ValueError('source architecture drift')
    setting = dict(cfg['model'], **cfg['architectures']['waveform'])
    setting.update(input_dim=400 if modality == 'eeg' else 40, channels=1 if modality == 'eeg' else 2,
                   sample_strides=[5, 4, 5] if modality == 'eeg' else [5])
    return model, setting


def diagnostics(model, batch, spec, wrapper=None):
    model.eval()
    with torch.inference_mode():
        if wrapper is None:
            result = model(batch['x'])
        else:
            prediction = wrapper(batch['x'].numpy())
            result = {key: torch.as_tensor(prediction[key]) for key in ('S', 'O', 'sum')}
            result['ids'] = {key: torch.as_tensor(value) for key, value in prediction['ids'].items()}
            result['commitment'] = batch['x'].new_zeros(())
        score, terms = _state_refinement_loss(result, batch, spec)
        replay = model.decode_ids(result['ids'])
    error = max(float((result[name] - replay[name]).abs().max()) for name in ('S', 'O'))
    if error != 0:
        raise AssertionError('integer replay differs')
    return dict(loss=float(score), terms={k: float(v) for k, v in terms.items()}, replay_max_abs=error)


def run(args, config):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / 'resolved_config.json'
    if resolved_path.exists() and json.loads(resolved_path.read_text()) != config:
        raise ValueError('contextual resolved configuration changed')
    write_json(resolved_path, config)
    semantic = config.get('assignment') is not None
    if semantic and args.modality != 'hb':
        raise ValueError('semantic assignment currently requires paired Hb')
    started = time.monotonic()
    torch.set_num_threads(config['resources']['torch_threads'])
    torch.manual_seed(config['training']['seed'])
    manifest, scales, _, _ = base_tokenizer_training_data(config, SOURCE)
    rows = manifest['rows']
    source_config = yaml.safe_load((SOURCE / config['source']['config']).read_text())
    if source_config['data'] != {**source_config['data'], **config['data']}:
        raise ValueError('source data boundary drift')
    source_candidate = config['source'].get('candidate', 'continuous')
    source_path = SOURCE / config['source']['run'] / 'models' / f'{args.modality}__waveform__{source_candidate}' / 'checkpoint_step_10000.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if (source['step'] != config['source']['checkpoint_step']
            or source['signature']['scales'] != manifest['scales_identity']
            or source['signature']['input'] != identity(manifest)
            or source['signature']['config'] != identity(source_config)
            or source['signature']['task'] != dict(modality=args.modality, architecture='waveform', candidate=source_candidate)):
        raise ValueError('fixed source checkpoint or scale drift')
    signature = dict(config=identity(config), input=identity(manifest), scales=identity(scales),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), source_signature=source['signature'])
    train = config['training']
    groups = base_tokenizer_sampling_index(rows, args.modality)
    cache = OrderedDict()
    rng = np.random.default_rng(train['seed'])
    examples = base_tokenizer_sample(rng, groups, train['dictionary_contexts'])
    if any(rows[i]['split'] != 'train' for i, _, _ in examples):
        raise ValueError('dictionary training boundary before arrays')
    mod_out = output / args.modality
    mod_out.mkdir(exist_ok=True)
    write_json(mod_out / 'training_support.json', dict(examples=examples, source=str(source_path),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_signature=source['signature'], source_step=source['step']))
    model, setting = configured_model(source_config, args.modality, 1024, source)
    latents_path = mod_out / 'training_latents.npz'
    if latents_path.exists():
        with np.load(latents_path) as arrays:
            if json.loads(str(arrays['signature'])) != signature:
                raise ValueError('training latent resume identity changed')
            latents = {key: arrays[key] for key in ('S', 'O')}
    else:
        encoded = {'S': [], 'O': []}
        model.eval()
        tick = time.monotonic()
        for offset in range(0, len(examples), 32):
            batch = _base_training_batch(rows, examples[offset:offset + 32], scales, 'cpu', cache, config)
            with torch.inference_mode():
                values = model.encode_continuous(batch['x'])
            for key, value in values.items():
                encoded[key].append(value.reshape(-1, value.shape[-1]).numpy())
            if offset % 256 == 0:
                write_json(mod_out / 'progress.json', dict(stage='encode_train', completed=offset + len(batch['x']),
                    total=len(examples), elapsed=time.monotonic()-tick))
        latents = {key: np.concatenate(value) for key, value in encoded.items()}
        latent_temporary = latents_path.with_suffix('.temporary.npz')
        np.savez_compressed(latent_temporary, **latents, signature=np.asarray(json.dumps(signature, sort_keys=True)))
        latent_temporary.replace(latents_path)
        write_json(mod_out / 'encoding.json', dict(seconds=time.monotonic()-tick, contexts=len(examples),
            patches=len(latents['S']), context_shape=list(batch['x'].shape), split='train'))
    features = {}
    feature_statistics = {}
    if semantic:
        model.eval()
        for branch in ('S', 'O'):
            decoded = []
            branch_latents = latents[branch].reshape(-1, 15, latents[branch].shape[-1])
            for offset in range(0, len(branch_latents), 32):
                with torch.inference_mode():
                    predicted = model._decode_component(branch, torch.as_tensor(branch_latents[offset:offset+32]))
                decoded.append(hb_semantic_features(predicted.numpy()).reshape(-1, 46))
            raw = np.concatenate(decoded)
            mean = raw.mean(axis=0).astype(np.float32)
            scale = np.maximum(raw.std(axis=0), 1e-6).astype(np.float32)
            weights = config['assignment']['weights']
            weight = np.r_[np.full(2, weights['level']/np.sqrt(2)),
                np.full(40, weights['shape']/np.sqrt(40)), np.full(2, weights['slope']/np.sqrt(2)),
                np.full(2, weights['curvature']/np.sqrt(2))].astype(np.float32)
            features[branch] = ((raw-mean)/scale*weight).astype(np.float32)
            feature_statistics.update({branch+'_mean':mean.astype(np.float32),
                branch+'_scale':scale.astype(np.float32), branch+'_weight':weight})
    diag_examples = base_tokenizer_sample(np.random.default_rng(923), groups, 64)
    diag_batch = _base_training_batch(rows, diag_examples, scales, 'cpu', cache, config)
    spec = source_config['architectures']['waveform']
    results = []
    for size in train['codebook_sizes']:
        if time.monotonic() - started > config['resources']['stage_wall_budget_seconds']:
            write_json(mod_out / 'budget.json', dict(status='budget_exhausted', elapsed=time.monotonic()-started))
            break
        job = mod_out / f'k{size}'
        job.mkdir(exist_ok=True)
        if (job / 'summary.json').exists():
            retained = json.loads((job / 'summary.json').read_text())
            if retained['metadata']['signature'] != signature:
                raise ValueError('completed contextual identity changed')
            results.append(retained)
            continue
        model, setting = configured_model(source_config, args.modality, size, source)
        reference = deepcopy(model).eval() if semantic else None
        assignment = dict(feature_statistics)
        metadata = dict(schema='contextual_latent_dictionary_v2' if semantic else 'contextual_latent_dictionary_v1', signature=signature, modality=args.modality, codebook_size=size,
            patches=15, patch_seconds=2, context_seconds=30, branch_bits=np.log2(size),
            bits_per_second_per_position_all_branches=float(np.log2(size)), source=str(source_path),
            source_step=10000, source_candidate=source_candidate, seed=train['seed'], quantization_levels=1,
            parameter_count_inference=sum(p.numel() for p in model.parameters()) * (2 if semantic else 1),
            source_historical_validation_exposure='all_original_validation_subjects',
            cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'])
        pre_path = job / 'preadapted.pt'
        if pre_path.exists():
            retained = torch.load(pre_path, map_location='cpu', weights_only=False)
            if retained['metadata'] != metadata:
                raise ValueError('preadapted contextual identity changed')
            model.load_state_dict(retained['model_state'])
            if semantic:
                assignment = retained['assignment']
                reference.load_state_dict(retained['reference_state'])
        else:
            for key in ('S', 'O'):
                tick = time.monotonic()
                write_json(mod_out / 'progress.json', dict(stage='kmeans', branch=key, k=size))
                fit = MiniBatchKMeans(n_clusters=size, random_state=train['seed'], batch_size=train['kmeans_batch'],
                    max_iter=train['kmeans_iterations'], n_init=1, init_size=max(size * 3, train['kmeans_batch']),
                    reassignment_ratio=.01).fit(features[key] if semantic else latents[key])
                q = model.branches[key]['quantizer']
                if semantic:
                    assignment[key+'_centers'] = fit.cluster_centers_.astype(np.float32)
                    count = np.bincount(fit.labels_, minlength=size)
                    sums = np.zeros((size, latents[key].shape[-1]), dtype=np.float64)
                    np.add.at(sums, fit.labels_, latents[key])
                    centers_array = sums / np.maximum(count[:, None], 1)
                    for empty in np.flatnonzero(count == 0):
                        closest = np.argmin(np.sum((features[key]-fit.cluster_centers_[empty])**2, axis=1))
                        centers_array[empty] = latents[key][closest]
                    centers = torch.as_tensor(centers_array, dtype=torch.float32)
                else:
                    centers = torch.as_tensor(fit.cluster_centers_, dtype=torch.float32)
                q.codebook.copy_(centers)
                q.ema_count.fill_(1.)
                q.ema_sum.copy_(centers)
                q.initialized.fill_(True)
                write_json(job / f'kmeans_{key}.json', dict(seconds=time.monotonic()-tick, inertia=float(fit.inertia_),
                    iterations=int(fit.n_iter_), occupied=int(len(np.unique(fit.labels_))),
                    patches=len(latents[key]), input_split='train'))
            extra = dict(assignment=assignment, reference_state=reference.state_dict()) if semantic else {}
            save(pre_path, dict(model_config=setting, model_state=model.state_dict(), metadata=metadata, step=0, **extra))
        extra = dict(assignment=assignment, reference_state=reference.state_dict()) if semantic else {}
        wrapper = SemanticContextualDictionary(model, metadata, reference, assignment) if semantic else None
        initial = diagnostics(model, diag_batch, spec, wrapper)
        for branch in model.branches.values():
            for name in ('input', 'encoder', 'to_latent', 'quantizer'):
                branch[name].requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=train['learning_rate'], weight_decay=train['weight_decay'])
        rng = np.random.default_rng(train['seed'] + 1000)
        step, history = 0, []
        checkpoint = job / 'checkpoint.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['metadata'] != metadata:
                raise ValueError('decoder continuation identity changed')
            model.load_state_dict(saved['model_state'])
            optimizer.load_state_dict(saved['optimizer_state'])
            rng.bit_generator.state = saved['rng']
            step, history = saved['step'], saved['history']
        tick = time.monotonic()
        for current in range(step, train['decoder_steps']):
            take = base_tokenizer_sample(rng, groups, train['batch_size'])
            batch = _base_training_batch(rows, take, scales, 'cpu', cache, config)
            model.eval()
            with torch.no_grad():
                if wrapper is None:
                    latent = model.encode_continuous(batch['x'])
                    ids = {key: model.branches[key]['quantizer']._assignment_logits(value.reshape(-1, value.shape[-1])).argmax(-1).reshape(value.shape[:2]) for key, value in latent.items()}
                else:
                    ids = {key: torch.as_tensor(value) for key, value in wrapper.encode(batch['x'].numpy()).items()}
                quantized = {key: model.branches[key]['quantizer'].get_embedding(value) for key, value in ids.items()}
            result = {key: model._decode_component(key, value) for key, value in quantized.items()}
            result.update(sum=result['S']+result['O'], commitment=batch['x'].new_zeros(()))
            optimizer.zero_grad()
            loss, terms = _state_refinement_loss(result, batch, spec)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite contextual training')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if current % 16 == 0:
                history.append(dict(step=current+1, loss=float(loss.detach()), elapsed=time.monotonic()-tick))
            if (current+1) % train['checkpoint_every'] == 0:
                save(checkpoint, dict(model_config=setting, model_state=model.state_dict(), metadata=metadata,
                    optimizer_state=optimizer.state_dict(), rng=rng.bit_generator.state, step=current+1, history=history, **extra))
                write_json(mod_out / 'progress.json', dict(stage='decoder_align', k=size, step=current+1,
                    total=train['decoder_steps'], seconds=time.monotonic()-tick))
        final = diagnostics(model, diag_batch, spec, wrapper)
        save(job / 'model.pt', dict(model_config=setting, model_state=model.state_dict(), metadata=metadata,
            step=train['decoder_steps'], **extra))
        wrapper = load_model(job / 'model.pt')
        wrapped = wrapper(diag_batch['x'].numpy())
        replay_ids = {key: value.copy() for key, value in wrapped['ids'].items()}
        np.savez_compressed(job / 'training_replay_ids.npz', **replay_ids)
        with np.load(job / 'training_replay_ids.npz') as retained:
            independently_decoded = load_model(job / 'model.pt').decode_ids(dict(retained))
        replay = max(float(np.max(np.abs(wrapped[key]-independently_decoded[key]))) for key in ('S', 'O', 'sum'))
        if replay != 0:
            raise AssertionError('saved integer replay differs')
        summary = dict(status='completed', initial_train_diagnostic=initial, final_train_diagnostic=final,
            model=str(job / 'model.pt'), metadata=metadata, train_replay_max_abs=replay,
            parameters=sum(p.numel() for p in model.parameters()), decoder_seconds=time.monotonic()-tick,
            history=history, validation_evaluated=False)
        write_json(job / 'summary.json', summary)
        results.append(summary)
    write_json(mod_out / 'summary.json', dict(status='completed' if len(results)==len(train['codebook_sizes']) else 'budget_exhausted', tasks=results, seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=SOURCE/'experiments/configs/physiology_semantic_tokenizer/dual_component_contextual_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modality', choices=['eeg', 'hb'], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    try:
        run(args, config)
    except BaseException as error:
        write_json(args.output / args.modality / 'failure.json', dict(type=type(error).__name__, reason=str(error)))
        raise


if __name__ == '__main__':
    main()
