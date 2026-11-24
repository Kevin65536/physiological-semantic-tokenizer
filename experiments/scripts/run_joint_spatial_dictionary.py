#!/usr/bin/env python3
"""Fit and audit a position-assigned single-ID spatial EEG dictionary."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import identity
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.analysis.spatial_tokenizer import completion, screen, write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg
from src.tokenizers.spatial_waveform_dictionary import (
    BANDS, SpatialWaveformDictionary, band_filter, load_model, spectral_features, joint_assignment_features,
)
from src.tokenizers.supervised_waveform_dictionary import load_model as load_independent

TRAIN_TARGETS = {}
INITIAL_ATOMS = {}


def fit_band(task):
    branch, band, cfg, output = task
    checkpoint = output / f'{branch}_band{band}.npz'
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as stored:
            if str(stored['config_identity']) != identity(cfg):
                raise ValueError('band checkpoint configuration changed')
        return str(checkpoint)
    started = time.monotonic()
    targets = TRAIN_TARGETS[branch]
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        count = cfg['model']['codebook_size'] - 1
        fraction = cfg['model'].get('semantic_distance_fraction')
        extra = {}
        if fraction is None:
            features = spectral_features(targets, band).astype(np.float32)
            initial = spectral_features(INITIAL_ATOMS[branch][:count], band).astype(np.float32)
        else:
            if not 0 < fraction < 1:
                raise ValueError('semantic distance fraction must be strictly between 0 and 1')
            raw = joint_assignment_features(targets, band)
            variance = np.var(raw, axis=0)
            carrier_weight = np.sqrt((1 - fraction) / max(variance[:-3].sum(), 1e-12))
            weights = np.r_[np.full(raw.shape[1]-3, carrier_weight),
                            np.sqrt(fraction / (3 * np.maximum(variance[-3:], 1e-12)))]
            features = (raw * weights).astype(np.float32)
            initial = (joint_assignment_features(INITIAL_ATOMS[branch][:count], band) * weights).astype(np.float32)
            extra['feature_weights'] = weights
        clustering = MiniBatchKMeans(n_clusters=count, init=initial, n_init=1,
            random_state=cfg['training']['seed'], batch_size=cfg['training']['kmeans_batch_size'],
            max_iter=cfg['training']['kmeans_max_iter'],
            max_no_improvement=cfg['training']['kmeans_max_no_improvement'],
            reassignment_ratio=cfg['training']['kmeans_reassignment_ratio'])
        labels = clustering.fit_predict(features)
        counts = np.bincount(labels, minlength=count)
        atoms = np.zeros((count + 1, 400), dtype=np.float32)
        np.add.at(atoms[1:], labels, targets)
        occupied = counts > 0
        atoms[1:][occupied] /= counts[occupied, None]
        atoms[1:][~occupied] = INITIAL_ATOMS[branch][:count][~occupied]
        if fraction is None:
            centers = spectral_features(atoms[1:], band).astype(np.float32)
        else:
            # A code represents the mean attributes of its members, including incoherent oscillations.
            centers = np.zeros((count, features.shape[1]), dtype=np.float64)
            np.add.at(centers, labels, features)
            centers[occupied] /= counts[occupied, None]
            centers[~occupied] = initial[~occupied]
            centers = centers.astype(np.float32)
        temporary = checkpoint.with_suffix('.pending.npz')
        np.savez_compressed(temporary, config_identity=identity(cfg), centers=centers, atoms=atoms,
                            counts=counts, iterations=clustering.n_iter_, **extra)
        temporary.replace(checkpoint)
        write(checkpoint.with_suffix('.json'), dict(status='completed', branch=branch, band_hz=BANDS[band],
            dictionary_training_patches=len(targets), occupied_codes=int(occupied.sum()),
            available_nonmissing_codes=count, reserved_missing_id=0,
            latent_dimensions=int(features.shape[1]), elapsed_seconds=time.monotonic() - started))
    return str(checkpoint)


def fit_spatial(model, rows, scales, plan, cfg, output):
    cache = OrderedDict()
    weights = {}
    for dataset, subjects in sorted(plan.items()):
        checkpoint = output / (dataset + '_weights.npz')
        if checkpoint.exists():
            with np.load(checkpoint) as stored:
                for branch in ('S', 'O'):
                    for band in range(len(BANDS)):
                        weights[dataset, branch, band] = stored[f'{branch}_{band}']
            continue
        names = model.metadata['positions'][dataset]
        roles = model.roles(dataset)
        subject_statistics = []
        for subject, crops in subjects.items():
            gram, cross = {}, {}
            counts = np.zeros(len(names))
            for branch in ('S', 'O'):
                for band in range(len(BANDS)):
                    sources = int(np.sum(roles == band))
                    gram[branch, band] = np.zeros((len(names), sources, sources))
                    cross[branch, band] = np.zeros((len(names), sources))
            for crop in crops:
                fields, observed, valid = base_tokenizer_joint_eeg(rows, crop, scales, names, cache)
                values = fields['x'].T.reshape(1, len(names), 15, 400)
                ids = model.encode(values, dataset, observed[None])
                independent = model.decode_ids(ids, dataset, spatial=False)
                counts += valid
                for branch in ('S', 'O'):
                    decoded = independent[branch][0].reshape(len(names), 6000).T
                    for band in range(len(BANDS)):
                        sources = band_filter(decoded, band, axis=0)[:, roles == band]
                        target = band_filter(fields[branch.lower()], band, axis=0)
                        g = sources.T @ sources / 6000
                        c = sources.T @ target / 6000
                        gram[branch, band][valid] += g
                        cross[branch, band][valid] += c[:, valid].T
            for key in gram:
                gram[key] /= np.maximum(counts, 1)[:, None, None]
                cross[key] /= np.maximum(counts, 1)[:, None]
            subject_statistics.append((gram, cross))
            write(output / 'spatial_progress.json', dict(dataset=dataset, subject=subject,
                fitted_subjects=len(subject_statistics), planned_subjects=len(subjects)))
        arrays = {}
        for branch in ('S', 'O'):
            for band in range(len(BANDS)):
                key = branch, band
                gram = np.mean([item[0][key] for item in subject_statistics], axis=0)
                cross = np.mean([item[1][key] for item in subject_statistics], axis=0)
                source_count = gram.shape[-1]
                strength = cfg['training']['spatial_ridge_fraction'] * np.trace(gram, axis1=1, axis2=2) / source_count + 1e-12
                diagonal = np.arange(source_count)
                gram[:, diagonal, diagonal] += strength[:, None]
                weights[dataset, branch, band] = np.linalg.solve(gram, cross[..., None])[..., 0].T
                arrays[f'{branch}_{band}'] = weights[dataset, branch, band]
        temporary = checkpoint.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **arrays)
        temporary.replace(checkpoint)
    return weights


def synthetic(output):
    rng = np.random.default_rng(53)
    metadata = dict(schema='spatial_waveform_dictionary_v1', codebook_size=4,
                    positions={'fixture': [str(i) for i in range(10)]})
    centers = {(branch, band): rng.normal(size=(3, spectral_features(np.zeros((1, 400)), band).shape[-1]))
               for branch in ('S', 'O') for band in range(len(BANDS))}
    atoms = {(branch, band): np.concatenate((np.zeros((1, 400)), rng.normal(size=(3, 400))))
             for branch in ('S', 'O') for band in range(len(BANDS))}
    weights = {}
    for branch in ('S', 'O'):
        for band in range(len(BANDS)):
            weights['fixture', branch, band] = np.zeros((2, 10))
            weights['fixture', branch, band][:, np.arange(10) % 5 == band] = np.eye(2)
    model = SpatialWaveformDictionary(metadata, {branch: np.full(3001, .5) for branch in ('S', 'O')}, centers, atoms, weights)
    values = rng.normal(size=(2, 10, 15, 400))
    observed = np.ones((2, 10), bool)
    observed[1, 0] = False
    ids = model.encode(values, 'fixture', observed)
    assert all(np.all(value[1, 0] == 0) for value in ids.values())
    checkpoint = output / 'synthetic_model.npz'
    model.save(checkpoint)
    restored = load_model(checkpoint)
    for spatial in (False, True):
        first = model.decode_ids(ids, 'fixture', spatial=spatial)
        replay = restored.decode_ids(ids, 'fixture', spatial=spatial)
        for branch in ('S', 'O', 'sum'):
            np.testing.assert_array_equal(first[branch], replay[branch])
        assert first['sum'].shape == values.shape and np.isfinite(first['sum']).all()
    try:
        model.decode_ids({branch: value.astype(float) for branch, value in ids.items()}, 'fixture')
    except ValueError:
        pass
    else:
        raise AssertionError('floating integer sequences accepted')
    invalid_ids = {branch: value.copy() for branch, value in ids.items()}
    invalid_ids['S'][0, 0, 0] = 4
    try:
        model.decode_ids(invalid_ids, 'fixture')
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-vocabulary ID accepted')
    write(output / 'synthetic.json', dict(status='passed', integer_only_replay='exact', missing_ID=0,
        shape=list(values.shape), branch_integer_sequences=2, IDs_per_2s_per_branch_position=1,
        decoder_continuous_inputs=0, out_of_vocabulary_and_float_inputs='rejected'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_dictionary_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--audit-output', type=Path)
    parser.add_argument('--ids-source', type=Path)
    parser.add_argument('--source-model', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.synthetic:
        synthetic(args.output)
        return
    cfg = yaml.safe_load(args.config.read_text())
    if (cfg['schema'] != 'dual_component_spatial_dictionary_v1' or cfg['training']['fitting_splits'] != ['train']
            or cfg['model']['context_s'] != 30 or cfg['model']['token_s'] != 2
            or cfg['model']['input'] != 'observed_X_only' or cfg['model']['missing_id'] != 0):
        raise ValueError('versioned X-only 30s single-ID spatial dictionary required')
    if (args.output / 'summary.json').exists():
        raise FileExistsError('retained completed model run')
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    source_path = ROOT / cfg['training']['initialization_checkpoint']
    source = load_independent(source_path)
    if source.metadata['input_identity'] != identity(manifest):
        raise ValueError('matched G0 source predictor required')
    example_record = json.loads((ROOT / cfg['training']['initialization_examples']).read_text())
    examples = example_record['examples']
    if any(rows[index]['split'] != 'train' for index, _, _ in examples):
        raise ValueError('train-only dictionary support before signal reads')
    plan = json.loads((ROOT / cfg['training']['spatial_plan']).read_text())['plan']
    if any(rows[crop['row']]['split'] != 'train' for subjects in plan.values() for crops in subjects.values() for crop in crops):
        raise ValueError('train-only spatial fit before signal reads')
    names = {dataset: sorted({name for crops in subjects.values() for crop in crops
                              for name in rows[crop['row']]['positions']}, key=str.upper) for dataset, subjects in plan.items()}
    metadata = dict(schema='spatial_waveform_dictionary_v1', codebook_size=cfg['model']['codebook_size'], positions=names,
        context_s=30, token_s=2, channels=1, input='X_only', branches=['S', 'O'], reserved_missing_ID=0,
        band_roles='canonical_native_electrode_index_modulo_5', bands_hz=BANDS,
        integer_count_per_branch_per_30s_position=15, bits_per_branch_per_position_second=np.log2(cfg['model']['codebook_size']) / 2,
        source_checkpoint=str(source_path), source_checkpoint_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        input_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'], config_identity=identity(cfg),
        seed=cfg['training']['seed'], validation_statistics_fit=0, decoder_continuous_side_information=0)
    if 'semantic_distance_fraction' in cfg['model']:
        metadata['semantic_distance_fraction'] = cfg['model']['semantic_distance_fraction']
    if cfg['model'].get('output_projection_hz') is not None:
        if cfg['model']['output_projection_hz'] != [1, 45]:
            raise ValueError('exact input-supported decoder projection required')
        metadata.update(output_projection_hz=[1, 45], projection_operator='30s_real_DFT_zero_phase_periodic_no_padding')
    if args.source_model is not None:
        if not args.evaluate_only or args.ids_source is None:
            raise ValueError('source model migration requires integer-only evaluation')
        metadata.update(projection_source_model=str(args.source_model),
                        projection_source_sha256=hashlib.sha256(args.source_model.read_bytes()).hexdigest())
    resolved = dict(config=cfg, metadata=metadata, dictionary_examples=examples, spatial_plan=plan)
    resolved_path = args.output / 'resolved_config.json'
    if resolved_path.exists() and json.loads(resolved_path.read_text()) != json.loads(json.dumps(resolved)):
        raise ValueError('retained run configuration or input identity changed')
    write(resolved_path, resolved)
    if args.evaluate_only:
        checkpoint = args.output / 'model.npz'
        if args.source_model is not None and not checkpoint.exists():
            model = load_model(args.source_model)
            if (model.metadata['input_identity'] != metadata['input_identity']
                    or model.metadata['scales_identity'] != metadata['scales_identity']
                    or model.metadata['positions'] != metadata['positions']):
                raise ValueError('same model data and native electrode support required')
            model.metadata = json.loads(json.dumps(metadata))
            model.save(checkpoint)
        model = load_model(checkpoint)
        if model.metadata != json.loads(json.dumps(metadata)):
            raise ValueError('frozen spatial model identity changed')
        audit_output = args.audit_output or args.output
        audit_output.mkdir(parents=True, exist_ok=True)
        if (audit_output / 'screen_summary.json').exists():
            raise FileExistsError('retained completed screen audit')
        started = time.monotonic()
        with threadpool_limits(cfg['resources']['numerical_threads_spatial']):
            summaries = screen(model, checkpoint, rows, scales, cfg, audit_output, args.ids_source)
        write(args.output / 'summary.json', completion(model, checkpoint, summaries, started, audit_output))
        return
    cache = OrderedDict()
    parts = [base_tokenizer_batch(rows, examples[begin:begin + 128], scales, array_cache=cache)
             for begin in range(0, len(examples), 128)]
    complete = np.concatenate([part['mask'] for part in parts]).all(axis=-1).reshape(-1)
    for branch in ('S', 'O'):
        TRAIN_TARGETS[branch] = np.concatenate([part[branch.lower()] for part in parts]).reshape(-1, 400)[complete]
        INITIAL_ATOMS[branch] = source.atoms[branch]
    del parts
    cache.clear()
    tasks = [(branch, band, cfg, args.output) for branch in ('S', 'O') for band in range(len(BANDS))]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['resources']['dictionary_workers'], mp_context=multiprocessing.get_context('fork')) as pool:
        list(pool.map(fit_band, tasks))
    centers, atoms, feature_weights = {}, {}, {}
    for branch, band, _, _ in tasks:
        with np.load(args.output / f'{branch}_band{band}.npz') as stored:
            centers[branch, band] = stored['centers'].copy()
            atoms[branch, band] = stored['atoms'].copy()
            if 'feature_weights' in stored:
                feature_weights[branch, band] = stored['feature_weights'].copy()
    model = SpatialWaveformDictionary(metadata, source.transfer, centers, atoms, {}, feature_weights=feature_weights)
    with threadpool_limits(cfg['resources']['numerical_threads_spatial']):
        model.weights = fit_spatial(model, rows, scales, plan, cfg, args.output)
        checkpoint = args.output / 'model.npz'
        model.save(checkpoint)
        summaries = screen(model, checkpoint, rows, scales, cfg, args.output)
    write(args.output / 'summary.json', completion(model, checkpoint, summaries, started, args.output))


if __name__ == '__main__':
    main()
