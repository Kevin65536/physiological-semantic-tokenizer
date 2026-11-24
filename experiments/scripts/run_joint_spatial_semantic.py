#!/usr/bin/env python3
"""Train complete two-second attribute/phase atoms and integer spatial decoding."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts import run_joint_spatial_temporal as temporal
from experiments.scripts.run_dual_component_tokenizer import identity, owner_lock
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.spatial_semantic_dictionary import SpatialSemanticDictionary, load_model, semantic_slot_features
from src.tokenizers.spectral_waveform_dictionary import band_components, raw_features
from src.tokenizers.supervised_waveform_dictionary import load_model as load_source

TRAIN = {}


def nearest(features, centers):
    norm = np.square(centers).sum(1)
    parts = []
    for start in range(0, len(features), 256):
        block = features[start:start+256]
        distance = np.square(block).sum(1, keepdims=True)+norm-2*block@centers.T
        parts.append(distance.argmin(1))
    return np.concatenate(parts)


def feature_coordinate(features, waveform_weight):
    mean = features.mean(0)
    scale = features.std(0).clip(1e-6)
    scale[:3] *= np.sqrt(3)
    # Preserve Euclidean phase/amplitude geometry inside the assigned waveform,
    # with its group weight calibrated on disjoint training subjects.
    scale[3:] = np.sqrt(np.mean(np.var(features[:, 3:], axis=0)))*np.sqrt(features.shape[1]-3)/waveform_weight
    scale[3:] = np.maximum(scale[3:], 1e-6)
    return mean, scale


def conditional_means(values, labels, count):
    totals = np.zeros((count, values.shape[1]), dtype=np.float64)
    np.add.at(totals, labels, values)
    occupancy = np.bincount(labels, minlength=count)
    totals /= np.maximum(occupancy[:, None], 1)
    totals[occupancy == 0] = values.mean(0)
    return totals, occupancy


def clustering(features, size, cfg, iterations):
    setting = cfg['training']
    return MiniBatchKMeans(n_clusters=size, random_state=setting['seed'], n_init=setting['kmeans_n_init'],
        batch_size=setting['kmeans_batch_size'], max_iter=iterations,
        max_no_improvement=setting['kmeans_max_no_improvement'],
        reassignment_ratio=setting['kmeans_reassignment_ratio'],
        init_size=max(3*size, setting['kmeans_batch_size'])).fit(features.astype(np.float32))


def fit_book(task):
    branch, slot, cfg, output = task
    folder = output/'books'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{branch}_{slot}.npz'
    if path.exists():
        with np.load(path) as archive:
            if str(archive['config_identity']) != identity(cfg):
                raise ValueError('retained complete atom codebook configuration changed')
        return branch, slot
    started = time.monotonic()
    prediction, target = TRAIN[branch]
    fit_indices, hold_indices = TRAIN['probe_indices']
    slot_samples = round(cfg['model']['slot_s']*200)
    with threadpool_limits(cfg['resources']['dictionary_threads_per_worker']):
        features = semantic_slot_features(prediction, slot, slot_samples)
        target_semantics = raw_features(target, None, None, 'theta_alpha_rms')
        target_slot = target[:, slot*slot_samples:(slot+1)*slot_samples]
        probe = []
        for weight in cfg['training']['waveform_weights']:
            mean, scale = feature_coordinate(features[fit_indices], weight)
            fit_features = (features[fit_indices]-mean)/scale
            fitted = clustering(fit_features, cfg['training']['probe_K'], cfg, cfg['training']['probe_kmeans_max_iter'])
            labels = nearest(fit_features, fitted.cluster_centers_)
            semantic_mean, _ = conditional_means(target_semantics[fit_indices], labels, cfg['training']['probe_K'])
            waveform_mean, _ = conditional_means(target_slot[fit_indices], labels, cfg['training']['probe_K'])
            held_labels = nearest((features[hold_indices]-mean)/scale, fitted.cluster_centers_)
            semantic_variance = np.var(target_semantics[fit_indices], axis=0).clip(1e-6)
            semantic_mse = np.mean((semantic_mean[held_labels]-target_semantics[hold_indices])**2, axis=0)/semantic_variance
            waveform_mse = np.mean((waveform_mean[held_labels]-target_slot[hold_indices])**2)/max(np.var(target_slot[fit_indices]), 1e-8)
            objective = .5*float(semantic_mse.mean())+.5*float(waveform_mse)
            probe.append(dict(waveform_weight=weight, semantic_normalized_MSE=semantic_mse,
                              slot_voltage_normalized_MSE=waveform_mse, objective=objective))
        chosen = min(probe, key=lambda record: (record['objective'], record['waveform_weight']))
        mean, scale = feature_coordinate(features, chosen['waveform_weight'])
        standardized = (features-mean)/scale
        fitted = clustering(standardized, cfg['training']['final_codebook_positive_atoms'], cfg,
                            cfg['training']['kmeans_max_iter'])
        labels = nearest(standardized, fitted.cluster_centers_)
        size = cfg['training']['final_codebook_positive_atoms']
        atoms, counts = conditional_means(target, labels, size)
        energy = np.zeros((size, 5)); envelope = np.zeros((size, 8))
        for start in range(0, len(target), 512):
            take = target[start:start+512]
            np.add.at(energy, labels[start:start+len(take)], np.mean(band_components(take)**2, axis=-1))
            np.add.at(envelope, labels[start:start+len(take)], np.sqrt(np.mean(take.reshape(-1, 8, 50)**2, axis=-1)))
        energy /= np.maximum(counts[:, None], 1)
        envelope /= np.maximum(counts[:, None], 1)
        arrays = dict(feature_mean=mean, feature_scale=scale, centers=fitted.cluster_centers_.astype(float),
                      atoms=np.concatenate((np.zeros((1, 400)), atoms)),
                      band_energy=np.concatenate((np.zeros((1, 5)), energy)),
                      rms_envelope=np.concatenate((np.zeros((1, 8)), envelope)), counts=np.r_[0, counts])
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, config_identity=np.asarray(identity(cfg)), **arrays)
        temporary.replace(path)
        temporal.write(folder/f'{branch}_{slot}.json', dict(status='completed', branch=branch, role=slot,
            feature_support=f'complete 2s attributes plus fixed {slot_samples/200:g}s phase waveform', prototype_samples=400,
            dictionary_patches=len(target), fit_probe_patches=len(fit_indices), held_probe_patches=len(hold_indices),
            probe=probe, selected_waveform_weight=chosen['waveform_weight'], final_positive_atoms=size,
            occupied=int(np.sum(counts > 0)), final_iterations=fitted.n_iter_, elapsed_seconds=time.monotonic()-started))
    return branch, slot


def probe_partition(examples, rows, cfg):
    subjects = {}
    for index, _, _ in examples:
        row = rows[index]
        subjects.setdefault(row['dataset'], set()).add(row['subject'])
    held_subjects = {}
    for dataset, values in subjects.items():
        ordered = sorted(values, key=lambda subject: hashlib.sha256(json.dumps([cfg['training']['seed'], dataset, subject]).encode()).hexdigest())
        held_subjects[dataset] = ordered[:max(1, len(ordered)//5)]
    held_windows = np.asarray([rows[index]['subject'] in held_subjects[rows[index]['dataset']] for index, _, _ in examples])
    return held_windows, held_subjects


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_semantic_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (cfg['schema'] != 'dual_component_spatial_semantic_v1' or not cfg['model']['interface_eligible']
            or cfg['model']['token_s'] != 2 or cfg['model']['slot_s'] not in (.25, .5) or cfg['model']['context_s'] != 30
            or cfg['model']['prototype_samples'] != 400 or cfg['model']['encoded_support_per_id_s'] != 2
            or cfg['model']['unencoded_decoder_inputs'] or cfg['evaluation']['role'] != 'screen'
            or cfg['training']['fitting_splits'] != ['train']):
        raise ValueError('full two-second X-only single-ID candidate contract required')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():
        raise FileExistsError('completed candidate evidence retained')
    with owner_lock(args.output):
        started = time.monotonic()
        manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
        rows = manifest['rows']
        source = load_source(ROOT/cfg['model']['source_checkpoint'])
        if source.metadata['input_identity'] != identity(manifest):
            raise ValueError('same frozen G0 source predictor required')
        support = json.loads((ROOT/cfg['training']['dictionary_support']).read_text())
        examples = support['examples']
        if any(rows[index]['split'] != 'train' or rows[index]['modality'] != 'eeg' for index, _, _ in examples):
            raise ValueError('train-only complete atom examples before signal reads')
        held_windows, held_subjects = probe_partition(examples, rows, cfg)
        cache = OrderedDict(); batches = []
        with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
            for start in range(0, len(examples), 128):
                batches.append(base_tokenizer_batch(rows, examples[start:start+128], scales, array_cache=cache, cache_limit=128))
            valid = np.concatenate([batch['mask'] for batch in batches]).all(axis=-1).reshape(-1)
            x = np.concatenate([batch['x'] for batch in batches])
            prediction = source.predict_continuous(x)
            held_patches = np.repeat(held_windows, 15)[valid]
            rng = np.random.default_rng(cfg['training']['seed'])
            fit_indices = rng.permutation(np.flatnonzero(~held_patches))[:cfg['training']['probe_fit_patches']]
            hold_indices = rng.permutation(np.flatnonzero(held_patches))[:cfg['training']['probe_holdout_patches']]
            TRAIN['probe_indices'] = fit_indices, hold_indices
            for branch in ('S', 'O'):
                target = np.concatenate([batch[branch.lower()] for batch in batches]).reshape(-1, 400)[valid]
                TRAIN[branch] = prediction[branch].reshape(-1, 400)[valid], target
        del batches, x, prediction
        cache.clear()
        plan = json.loads((ROOT/cfg['training']['spatial_plan']).read_text())['plan']
        if any(rows[crop['row']]['split'] != 'train' for subjects in plan.values() for crops in subjects.values() for crop in crops):
            raise ValueError('train-only spatial map fitting required')
        with np.load(ROOT/cfg['model']['namespace_checkpoint']) as archive:
            namespace = json.loads(str(archive['metadata']))
        slot_samples = round(cfg['model']['slot_s']*200)
        slot_count = 400//slot_samples
        metadata = dict(schema='spatial_semantic_dictionary_v1', codebook_size=cfg['model']['emitted_codebook_size'],
            positions=namespace['positions'], slot_samples=slot_samples, context_s=30, token_s=2,
            encoded_support_per_ID_seconds=2, prototype_samples=400, reserved_missing_ID=0,
            interface_eligible=True, input='X_only', integer_count_per_branch_position_window=15,
            fixed_length_bits_per_ID=11, cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
            config_identity=identity(cfg), calibration_gain_cap=cfg['model']['calibration_gain_cap'],
            local_namespace='fixed_native_position_role_and_branch_and_ID',
            roles=f'fixed_native_electrode_index_modulo_{slot_count}',
            feature_groups=dict(full_2s_theta_alpha_RMS=3, assigned_slot_waveform=slot_samples),
            source_checkpoint=str(ROOT/cfg['model']['source_checkpoint']),
            source_checkpoint_sha256=hashlib.sha256((ROOT/cfg['model']['source_checkpoint']).read_bytes()).hexdigest())
        temporal.write(args.output/'resolved_config.json', dict(config=cfg, metadata=metadata, training_plan=plan,
            held_train_subjects_for_weight_probe=held_subjects, probe_source_predictor_exposure='frozen original predictor already used all original train subjects',
            complete_dictionary_patches=int(valid.sum()), dictionary_support=cfg['training']['dictionary_support']))
        shutil.copy2(__file__, args.output/'source_execution.py')
        shutil.copy2(ROOT/'src/tokenizers/spatial_semantic_dictionary.py', args.output/'model_source.py')
        tasks = [(branch, slot, cfg, args.output) for branch in ('S', 'O') for slot in range(slot_count)]
        with ProcessPoolExecutor(max_workers=cfg['resources']['dictionary_workers'], mp_context=multiprocessing.get_context('fork')) as pool:
            for future in as_completed([pool.submit(fit_book, task) for task in tasks]):
                print(json.dumps(dict(stage='book_completed', book=future.result())), flush=True)
        books = {}
        for branch, slot, _, _ in tasks:
            with np.load(args.output/'books'/f'{branch}_{slot}.npz') as archive:
                books[branch, slot] = {key: archive[key] for key in archive.files if key != 'config_identity'}
        blank_weights = {(dataset, mode, branch, slot): np.zeros((int(np.sum(np.arange(len(names)) % slot_count == slot)), len(names)))
                         for dataset, names in metadata['positions'].items() for mode in ('continuous', 'quantized')
                         for branch in ('S', 'O') for slot in range(slot_count)}
        model = SpatialSemanticDictionary(metadata, source, books, blank_weights)
        unfitted = args.output/'atoms_model.npz'
        model.save(unfitted)
        temporal.TRAIN_ROWS, temporal.TRAIN_SCALES = rows, scales
        dataset_tasks = [(dataset, subjects, cfg, metadata, args.output, unfitted) for dataset, subjects in plan.items()]
        with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers'], mp_context=multiprocessing.get_context('fork')) as pool:
            for future in as_completed([pool.submit(temporal.fit_dataset, task) for task in dataset_tasks]):
                print(json.dumps(dict(stage='spatial_fit_completed', dataset=future.result())), flush=True)
        weights = {}
        for dataset in plan:
            with np.load(args.output/dataset/'weights.npz') as archive:
                for mode in ('continuous', 'quantized'):
                    for branch in ('S', 'O'):
                        for slot in range(slot_count):
                            weights[dataset, mode, branch, slot] = archive[f'{mode}_{branch}_{slot}']
        model.weights = weights
        model.save(args.output/'model.npz')
        with threadpool_limits(cfg['resources']['screen_threads']):
            summaries = temporal.screen(model, cfg, rows, scales, args.output, model_loader=load_model)
        temporal.write(args.output/'summary.json', dict(status='completed', metadata=metadata,
            elapsed_seconds=time.monotonic()-started, checkpoint='model.npz',
            screen_scores={mode: result['selection_score'] for mode, result in summaries.items()},
            screen_evidence='screen_summary.json', interface_eligible=True,
            decoder_information='saved S/O integer sequences and frozen model with native position metadata only'))


if __name__ == '__main__':
    main()
