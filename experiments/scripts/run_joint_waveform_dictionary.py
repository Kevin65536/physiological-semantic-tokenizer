#!/usr/bin/env python3
"""Fit an auditable X-only continuous predictor and single-ID waveform dictionary."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import yaml

from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_sample, base_tokenizer_sampling_index
from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary, load_model
from experiments.scripts.run_dual_component_tokenizer import identity
from src.data.dual_component_tokenizer import base_tokenizer_training_data


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_config(cfg):
    if cfg.get('schema') != 'dual_component_joint_dictionary_v1':
        raise ValueError('versioned dictionary configuration required')
    model, training = cfg['model'], cfg['training']
    if (model['context_s'] != 30 or model['token_s'] != 2
            or model['channels'] != {'eeg': 1, 'hb': 2}
            or model['quantizer'] != 'one_nearest_atom_per_branch_patch'
            or model['decoder'] != 'independently_index_frozen_two_second_waveform_atoms'
            or model['unencoded_per_window_decoder_inputs']):
        raise ValueError('30-second X, one two-second integer, joint Hb, fixed-atom decoder contract required')
    sizes = model['codebook_sizes']
    if (not sizes or any(type(size) is not int or size < 2 for size in sizes)
            or len(set(sizes)) != len(sizes)
            or type(training['windows_per_modality']) is not int
            or training['windows_per_modality'] * 15 < max(sizes)):
        raise ValueError('valid distinct vocabularies and sufficient training patch budget required')
    for modality, dimension in [('eeg', 400), ('hb', 40)]:
        rank = model['output_pca_dim'][modality]
        if type(rank) is not int or not 1 <= rank <= dimension:
            raise ValueError('PCA rank must fit the native waveform patch dimension')
    if not 0 < model['frequency_ridge_fraction'] < 1:
        raise ValueError('positive bounded frequency regularization required')
    if training['validation_selection'] != 'forbidden_in_training_entry':
        raise ValueError('train-only producer required')


def preserve_record(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('retained run identity or training support changed: ' + str(path))
        return
    write_json(path, payload)


def fit_frequency(batch, channels, ridge_fraction):
    complete = batch['mask'].all(axis=(1, 2))
    if not complete.any():
        raise ValueError('no complete supervised training context')
    shape = (int(complete.sum()), -1, channels)
    spectrum = np.fft.rfft(batch['x'][complete].reshape(shape), axis=1)
    gram = np.einsum('nfi,nfj->fij', spectrum.conj(), spectrum, optimize=True)
    energy = np.trace(gram, axis1=1, axis2=2).real / channels
    ridge = ridge_fraction * energy + 1e-12
    regularized = gram + ridge[:, None, None] * np.eye(channels)[None]
    transfer = {}
    for branch in ('S', 'O'):
        target = np.fft.rfft(batch[branch.lower()][complete].reshape(shape), axis=1)
        cross = np.einsum('nfi,nfj->fij', spectrum.conj(), target, optimize=True)
        transfer[branch] = np.linalg.solve(regularized, cross)
    return transfer, int(complete.sum())


def fit_modality(cfg, rows, scales, modality, out):
    start = time.monotonic()
    setting = cfg['training']
    rng = np.random.default_rng(setting['seed'])
    examples = base_tokenizer_sample(rng, base_tokenizer_sampling_index(rows, modality), setting['windows_per_modality'])
    if any(rows[index]['split'] != 'train' for index, _, _ in examples):
        raise ValueError('training-only examples before signal reads')
    cache = OrderedDict()
    parts = []
    for offset in range(0, len(examples), 128):
        parts.append(base_tokenizer_batch(rows, examples[offset:offset + 128], scales, array_cache=cache,
                                         cache_limit=cfg['resources']['mmap_parent_cache']))
    batch = {key: np.concatenate([part[key] for part in parts]) for key in ('x', 's', 'o', 'mask')}
    del parts
    channels = cfg['model']['channels'][modality]
    transfer, frequency_windows = fit_frequency(batch, channels, cfg['model']['frequency_ridge_fraction'])
    dimension = batch['x'].shape[-1]
    keep = batch['mask'].all(axis=-1).reshape(-1)
    bases, means, projected, targets = {}, {}, {}, {}
    for branch in ('S', 'O'):
        values = batch[branch.lower()].reshape(-1, dimension)[keep].astype(np.float64)
        means[branch] = values.mean(axis=0)
        covariance = np.cov(values, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        rank = cfg['model']['output_pca_dim'][modality]
        bases[branch] = eigenvectors[:, -rank:][:, ::-1].T
        projected[branch] = ((values - means[branch]) @ bases[branch].T).astype(np.float32)
        targets[branch] = values
    folder = out / modality
    folder.mkdir(exist_ok=True)
    preserve_record(folder / 'training_examples.json', dict(examples=[list(item) for item in examples], frequency_complete_windows=frequency_windows,
        dictionary_complete_patches=int(keep.sum()), planned_windows=len(examples), input_identity=cfg['_input_identity'],
        sampling='equal_dataset_subject_record_crop_position', seed=setting['seed'], validation_arrays_read=0,
        complete_patch_rule='all G0 source scoring mask entries true; excluded entries remain in planned denominator'))
    history = []
    for size in cfg['model']['codebook_sizes']:
        destination = folder / ('k' + str(size) + '.npz')
        if destination.exists():
            model = load_model(destination)
            if (model.metadata['config_identity'] != cfg['_config_identity']
                    or model.metadata['input_identity'] != cfg['_input_identity']
                    or model.modality != modality or model.codebook_size != size):
                raise ValueError('existing checkpoint belongs to another configuration')
            continue
        stage_start = time.monotonic()
        atoms = {}
        fitting = {}
        for branch in ('S', 'O'):
            clustering = MiniBatchKMeans(n_clusters=size, random_state=setting['seed'],
                n_init=setting['kmeans_n_init'], max_iter=setting['kmeans_max_iter'],
                max_no_improvement=setting['kmeans_max_no_improvement'],
                batch_size=setting['kmeans_batch_size'], reassignment_ratio=setting['kmeans_reassignment_ratio'],
                init_size=max(size * 3, setting['kmeans_batch_size']))
            labels = clustering.fit_predict(projected[branch])
            counts = np.bincount(labels, minlength=size)
            sums = np.zeros((size, dimension), dtype=np.float64)
            np.add.at(sums, labels, targets[branch])
            valid = counts > 0
            sums[valid] /= counts[valid, None]
            sums[~valid] = means[branch] + clustering.cluster_centers_[~valid] @ bases[branch]
            atoms[branch] = sums
            fitting[branch] = dict(iterations=clustering.n_iter_, inertia=float(clustering.inertia_),
                occupied_training_clusters=int(valid.sum()), planned_clusters=size,
                atom_parameters=int(sums.size), pca_dimensions=int(bases[branch].shape[0]))
        metadata = dict(schema='supervised_waveform_dictionary_v1', modality=modality, codebook_size=size,
            input_dim=dimension, context_s=30, token_s=2, input='X_only', decoder='frozen_atoms_only',
            integers_per_30s_per_position=30, bits_per_branch_per_position_per_second=float(np.log2(size) / 2),
            fixed_length_bits_per_30s_per_position=int(30 * np.ceil(np.log2(size))),
            seed=setting['seed'], config_identity=cfg['_config_identity'], input_identity=cfg['_input_identity'],
            fitting=fitting, frequency_complete_training_windows=frequency_windows,
            time_utc=datetime.now(timezone.utc).isoformat(), validation_arrays_read=0)
        model = SupervisedWaveformDictionary(metadata, transfer, means, bases, atoms)
        model.save(destination)
        check = batch['x'][:8]
        ids = model.encode(check)
        prediction = model.decode_ids(ids)
        replay = load_model(destination).decode_ids(ids)
        maximum = max(float(np.max(np.abs(prediction[branch] - replay[branch]))) for branch in ('S', 'O'))
        if maximum != 0:
            raise ValueError('integer-only checkpoint replay differs')
        train_diagnostic = {}
        for branch in ('S', 'O'):
            truth = batch[branch.lower()][:128]
            mask = batch['mask'][:128]
            target_ids = model.nearest_ids(truth, branch)
            target_recon = model.atoms[branch][target_ids]
            continuous = model.predict_continuous(batch['x'][:128])[branch]
            hard = model(batch['x'][:128])[branch]
            variance = float(np.mean(np.square(truth[mask] - truth[mask].mean())))
            train_diagnostic[branch] = {name: float(np.mean(np.square(values[mask] - truth[mask])) / variance)
                for name, values in [('continuous_X', continuous), ('hard_X', hard), ('target_input_dictionary', target_recon)]}
        record = dict(checkpoint=str(destination), elapsed_seconds=time.monotonic() - stage_start,
                      training_only_diagnostic_NMSE=train_diagnostic, integer_replay_max_abs=maximum,
                      diagnostic_denominator='variance of scored target elements across first128 training windows; not G0 macro',
                      fitting=fitting)
        history.append(record)
        write_json(folder / ('k' + str(size) + '_training.json'), record)
        write_json(out / 'progress.json', dict(status='training', modality=modality, codebook_size=size,
                                              completed=history, elapsed_seconds=time.monotonic() - start))
        print(json.dumps(record), flush=True)
    return dict(modality=modality, seconds=time.monotonic() - start, completed_codebook_sizes=cfg['model']['codebook_sizes'])


def synthetic():
    rng = np.random.default_rng(3)
    metadata = dict(schema='supervised_waveform_dictionary_v1', modality='hb', codebook_size=4)
    transfer = {branch: np.tile(np.eye(2)[None] * .5, (151, 1, 1)).astype(complex) for branch in ('S', 'O')}
    means = {branch: np.zeros(40) for branch in ('S', 'O')}
    bases = {branch: np.eye(40) for branch in ('S', 'O')}
    atoms = {branch: rng.normal(size=(4, 40)) for branch in ('S', 'O')}
    model = SupervisedWaveformDictionary(metadata, transfer, means, bases, atoms)
    values = rng.normal(size=(2, 15, 40))
    output = model(values)
    replay = model.decode_ids(output['ids'])
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(output[branch], replay[branch])
    np.testing.assert_allclose(model.predict_continuous(values)['sum'], values, atol=1e-12)
    try:
        model.decode_ids({'S': np.zeros((2, 15)), 'O': np.zeros((2, 15))})
    except ValueError:
        pass
    else:
        raise AssertionError('floating IDs accepted')
    return dict(status='passed', roundtrip='exact', input_shape=[2, 15, 40], integer_sequences=2, decoder_continuous_inputs=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/configs/physiology_semantic_tokenizer/dual_component_joint_dictionary_v1.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--modality', choices=['eeg', 'hb'])
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        write_json(out / 'synthetic.json', synthetic())
        return
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    validate_config(cfg)
    modalities = [args.modality] if args.modality else ['eeg', 'hb']
    if args.modality:
        cfg['_requested_modality'] = args.modality
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    cfg['_input_identity'] = identity(manifest)
    cfg['_config_identity'] = identity(cfg)
    preserve_record(out / 'resolved_config.json', cfg)
    if (out / 'summary.json').exists():
        for modality in modalities:
            for size in cfg['model']['codebook_sizes']:
                retained = load_model(out / modality / ('k' + str(size) + '.npz'))
                if (retained.metadata['config_identity'] != cfg['_config_identity']
                        or retained.metadata['input_identity'] != cfg['_input_identity']
                        or retained.modality != modality or retained.codebook_size != size):
                    raise ValueError('completed dictionary checkpoint identity changed')
        return
    with threadpool_limits(limits=cfg['resources']['numerical_threads']):
        results = [fit_modality(cfg, manifest['rows'], scales, modality, out) for modality in modalities]
    write_json(out / 'summary.json', dict(status='trained_pending_independent_evaluation', models=results,
                                         validation_arrays_read=0, test_arrays_read=0))


if __name__ == '__main__':
    main()
