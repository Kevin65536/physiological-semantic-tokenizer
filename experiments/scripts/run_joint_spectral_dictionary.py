#!/usr/bin/env python3
"""Train matched single-ID EEG dictionaries with spectral dynamics assignments."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import identity, owner_lock
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from experiments.scripts.run_joint_waveform_dictionary import preserve_record, write_json
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.tokenizers.supervised_waveform_dictionary import load_model as load_source
from src.tokenizers.spectral_waveform_dictionary import (
    BANDS, SpectralWaveformDictionary, calibrated_atoms, load_model, raw_features,
)


def validate(cfg):
    model, training = cfg['model'], cfg['training']
    if (cfg['schema'] not in ('dual_component_joint_spectral_v1', 'dual_component_joint_spectral_v2') or model['context_s'] != 30
            or model['token_s'] != 2 or model['input'] != 'observed_X_only'
            or model['modality'] != 'eeg' or model['quantization_levels'] != 1
            or model['unencoded_per_window_decoder_inputs'] or model['decoder'] != 'fixed_atom_lookup'
            or model['feature_bands_hz'] != [list(band) for band in BANDS]
            or model['local_rms_samples'] != 50
            or training['validation_selection'] != 'forbidden_in_training_entry'):
        raise ValueError('fixed X-only 30-second single-ID EEG contract required')
    minimum_rank = 0 if model.get('feature_mode') == 'theta_alpha_rms' else 1
    if not minimum_rank <= model['waveform_pca_dim'] <= 400 or model['codebook_size'] < 2:
        raise ValueError('valid phase-sensitive rank and vocabulary required')


def fit_branch(branch, prediction, target, cfg, folder, signature=None):
    began = time.monotonic()
    model, training = cfg['model'], cfg['training']
    destination = folder / ('fit_' + branch + '.npz')
    if destination.exists():
        with np.load(destination) as retained:
            if json.loads(str(retained['signature'])) != signature:
                raise ValueError('retained branch config, input or source signature changed')
            return branch, {key: retained[key] for key in retained.files if key != 'signature'}
    mean = prediction.mean(axis=0)
    rank = model['waveform_pca_dim']
    if rank:
        _, vectors = np.linalg.eigh(np.cov(prediction, rowvar=False))
        basis = vectors[:, -rank:][:, ::-1].T
    else:
        basis = np.empty((0, 400), dtype=np.float64)
    mode = model.get('feature_mode', 'dynamics_waveform')
    features = raw_features(prediction, mean, basis, mode)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0).clip(1e-6)
    if mode == 'dynamics_waveform':
        feature_scale[:40] *= np.sqrt(40) / model['dynamics_weight']
        feature_scale[40:] *= np.sqrt(len(feature_scale) - 40) / model['waveform_weight']
    elif mode == 'semantic_waveform':
        feature_scale[:3] *= np.sqrt(3) / model['dynamics_weight']
        feature_scale[3:] *= np.sqrt(rank) / model['waveform_weight']
    standardized = ((features - feature_mean) / feature_scale).astype(np.float32)
    size = model['codebook_size']
    clustering = MiniBatchKMeans(n_clusters=size, random_state=training['seed'], n_init=training['kmeans_n_init'],
        max_iter=training['kmeans_max_iter'], batch_size=training['kmeans_batch_size'],
        max_no_improvement=training['kmeans_max_no_improvement'],
        reassignment_ratio=training['kmeans_reassignment_ratio'], init_size=max(size * 3, training['kmeans_batch_size']))
    labels = clustering.fit_predict(standardized)
    count = np.bincount(labels, minlength=size)
    atoms = np.zeros((size, 400), dtype=np.float64)
    np.add.at(atoms, labels, target)
    atoms /= np.maximum(count[:, None], 1)
    distance = np.sum((standardized - clustering.cluster_centers_[labels]) ** 2, axis=-1)
    # One nearest training member per occupied code supplies auditable fallback phases.
    representative_index = np.full(size, -1, dtype=np.int64)
    for index in np.argsort(distance)[::-1]:
        representative_index[labels[index]] = index
    empty = representative_index < 0
    representative_index[empty] = 0
    representatives = target[representative_index]
    atoms[empty] = representatives[empty]
    energy_atoms, energy, fallback = calibrated_atoms(target, labels, atoms, representatives, size)
    energy_atoms[empty] = atoms[empty]
    result = dict(means=mean, bases=basis, feature_mean=feature_mean, feature_scale=feature_scale,
        centers=clustering.cluster_centers_.astype(np.float64), mean_atoms=atoms, energy_atoms=energy_atoms,
        training_count=count, conditional_band_energy=energy, phase_fallback=fallback,
        representative_patch_index=representative_index, labels=labels)
    temporary = destination.with_suffix('.pending.npz')
    np.savez_compressed(temporary, signature=np.asarray(json.dumps(signature, sort_keys=True)), **result)
    temporary.replace(destination)
    write_json(folder / ('fit_' + branch + '.json'), dict(branch=branch, seconds=time.monotonic() - began,
        train_patches=len(target), codebook_size=size, occupied=int((count > 0).sum()),
        phase_fallback_bands=int(fallback.sum()), iterations=clustering.n_iter_, inertia=float(clustering.inertia_)))
    return branch, result


def run(cfg, out):
    validate(cfg)
    out.mkdir(parents=True, exist_ok=True)
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    source = load_source(ROOT / cfg['source']['model'])
    if source.metadata['input_identity'] != identity(manifest):
        raise ValueError('matched dictionary G0 identity drift before arrays')
    support = json.loads((ROOT / cfg['source']['support']).read_text())
    examples = support['examples']
    if len(examples) != cfg['training']['windows'] or support['seed'] != cfg['training']['seed']:
        raise ValueError('matched exact training support and seed required')
    rows = manifest['rows']
    if any(rows[i]['split'] != 'train' or rows[i]['modality'] != 'eeg' for i, _, _ in examples):
        raise ValueError('train-only EEG support before signal reads')
    preserve_record(out / 'resolved_config.json', cfg)
    preserve_record(out / 'training_examples.json', support)
    source_identity = dict(config=identity(cfg), input=identity(manifest),
        source_sha256=hashlib.sha256((ROOT / cfg['source']['model']).read_bytes()).hexdigest(),
        support_sha256=hashlib.sha256((ROOT / cfg['source']['support']).read_bytes()).hexdigest())
    summary_path = out / 'summary.json'
    if summary_path.exists():
        completed = json.loads(summary_path.read_text())
        paths = [Path(record['checkpoint']) for record in completed['models']]
        for path in paths:
            fitted = load_model(path)
            if fitted.metadata['config_identity'] != identity(cfg) or fitted.metadata['input_identity'] != identity(manifest):
                raise ValueError('completed model configuration or input changed')
            if any(not np.array_equal(fitted.transfer[branch], source.transfer[branch]) for branch in ('S', 'O')):
                raise ValueError('completed model frequency predictor changed')
        if not (out / 'source_identity.json').exists():
            preserve_record(out / 'legacy_source_identity_audit.json', dict(
                recorded_after_completion=True, original_source_byte_hash_available=False,
                used_transfer_arrays_match_current_source_exactly=True,
                preserved_models=[str(path) for path in paths], source_identity=source_identity))
        preserve_record(out / 'source_identity.json', source_identity)
        preserve_record(out / 'completion_identity.json', dict(summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            checkpoints={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
        return completed
    preserve_record(out / 'source_identity.json', source_identity)
    began = time.monotonic()
    cache, parts = OrderedDict(), []
    for offset in range(0, len(examples), 128):
        parts.append(base_tokenizer_batch(rows, examples[offset:offset + 128], scales, array_cache=cache,
                                         cache_limit=cfg['resources']['mmap_parent_cache']))
    batch = {key: np.concatenate([part[key] for part in parts]) for key in ('x', 's', 'o', 'mask')}
    del parts
    prediction = source.predict_continuous(batch['x'])
    keep = batch['mask'].all(axis=-1).reshape(-1)
    folder = out / 'eeg'
    folder.mkdir(exist_ok=True)
    write_json(out / 'progress.json', dict(stage='feature_clustering', train_windows=len(examples),
        complete_patches=int(keep.sum()), planned_patches=int(keep.size), validation_arrays_read=0))
    fitted = {}
    with threadpool_limits(limits=cfg['resources']['numerical_threads_per_worker']):
        with ProcessPoolExecutor(max_workers=cfg['resources']['branch_workers'],
                                 mp_context=multiprocessing.get_context('spawn')) as workers:
            jobs = [workers.submit(fit_branch, branch, prediction[branch].reshape(-1, 400)[keep],
                batch[branch.lower()].reshape(-1, 400)[keep], cfg, folder, source_identity) for branch in ('S', 'O')]
            for job in as_completed(jobs):
                branch, result = job.result()
                fitted[branch] = result
                write_json(out / 'progress.json', dict(stage='feature_clustering', completed_branches=list(fitted),
                    seconds=time.monotonic() - began, validation_arrays_read=0))
    models = []
    for mode in ('mean', 'energy'):
        metadata = dict(schema='spectral_waveform_dictionary_v1', modality='eeg', codebook_size=cfg['model']['codebook_size'],
            config_identity=identity(cfg), cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
            input_identity=identity(manifest), context_s=30, token_s=2, input='X_only', decoder='frozen_atoms_only',
            source_sha256=source_identity['source_sha256'], feature_mode=cfg['model'].get('feature_mode', 'dynamics_waveform'),
            atom_mode=mode, frequency_predictor=str(ROOT / cfg['source']['model']), seed=cfg['training']['seed'],
            feature_groups=(dict(theta_alpha_RMS=3, waveform_PCA=cfg['model']['waveform_pca_dim'])
                if cfg['model'].get('feature_mode') in ('theta_alpha_rms', 'semantic_waveform')
                else dict(log_band_RMS_8_quarter_second_blocks=40, waveform_PCA=cfg['model']['waveform_pca_dim'])),
            integers_per_30s_per_position=30, bits_per_branch_per_position_per_second=np.log2(cfg['model']['codebook_size']) / 2,
            validation_arrays_read=0, test_arrays_read=0, time_utc=datetime.now(timezone.utc).isoformat(),
            phase_interpretation='fixed training atom phases; event energy and envelope are quantized, phase recovery is evaluated separately')
        fields = {name: {branch: fitted[branch][name] for branch in ('S', 'O')}
                  for name in ('means', 'bases', 'feature_mean', 'feature_scale', 'centers')}
        model = SpectralWaveformDictionary(metadata, source.transfer, atoms={branch: fitted[branch][mode + '_atoms']
            for branch in ('S', 'O')}, **fields)
        path = folder / f"k{cfg['model']['codebook_size']}_{mode}.npz"
        if path.exists():
            existing = load_model(path)
            keys = ('config_identity', 'input_identity', 'source_sha256', 'atom_mode')
            if any(existing.metadata[key] != metadata[key] for key in keys):
                raise ValueError('retained frozen checkpoint identity changed')
            for name in ('atoms', 'means', 'bases', 'feature_mean', 'feature_scale', 'centers', 'transfer'):
                if any(not np.array_equal(getattr(existing, name)[branch], getattr(model, name)[branch]) for branch in ('S', 'O')):
                    raise ValueError('retained frozen checkpoint arrays changed')
            model = existing
        else:
            model.save(path)
        check = model(batch['x'][:8])
        replay = load_model(path).decode_ids(check['ids'])
        difference = max(float(np.max(np.abs(check[key] - replay[key]))) for key in ('S', 'O', 'sum'))
        if difference != 0:
            raise AssertionError('frozen integer replay differs')
        models.append(dict(checkpoint=str(path), mode=mode, integer_replay_max_abs=difference))
    write_json(out / 'summary.json', dict(status='trained_pending_screen', models=models,
        seconds=time.monotonic() - began, validation_arrays_read=0, test_arrays_read=0,
        matched_support=str(ROOT / cfg['source']['support']), complete_training_patches=int(keep.sum())))
    preserve_record(out / 'completion_identity.json', dict(summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        checkpoints={record['checkpoint']: hashlib.sha256(Path(record['checkpoint']).read_bytes()).hexdigest() for record in models}))


def diagnose_pareto(cfg, out):
    """Select atom geometry on training-subject holdout with a fixed X-only front."""
    from src.analysis.dual_component_tokenizer import base_training_metrics, base_training_aggregate
    from src.analysis.tokenizer_pattern_audit import GroupTotals
    validate(cfg)
    out.mkdir(parents=True, exist_ok=True)
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    source = load_source(ROOT / cfg['source']['model'])
    if source.metadata['input_identity'] != identity(manifest):
        raise ValueError('fixed source input identity changed')
    support = json.loads((ROOT / cfg['source']['support']).read_text())
    examples = np.asarray(support['examples'], dtype=np.int64)
    if len(examples) != cfg['training']['windows'] or support['seed'] != cfg['training']['seed']:
        raise ValueError('matched training support and seed required for component CV')
    rows = manifest['rows']
    if any(rows[i]['split'] != 'train' or rows[i]['modality'] != 'eeg' for i, _, _ in examples):
        raise ValueError('training-subject diagnostic before arrays')
    subjects = {}
    for i, _, _ in examples:
        row = rows[i]
        subjects.setdefault(row['dataset'], set()).add(row['subject'])
    held = set()
    for dataset, values in subjects.items():
        ordered = sorted(values, key=lambda subject: hashlib.sha256(
            f'joint-atom-cv-v1/{dataset}/{subject}'.encode()).digest())
        held.update((dataset, subject) for subject in ordered[:max(1, len(ordered)//5)])
    selection = np.asarray([(rows[i]['dataset'], rows[i]['subject']) in held for i, _, _ in examples])
    if not selection.any() or selection.all():
        raise ValueError('separate atom fitting and component-selection subjects required')
    preserve_record(out / 'resolved_config.json', cfg)
    preserve_record(out / 'fold.json', dict(fit_examples=examples[~selection].tolist(),
        heldout_examples=examples[selection].tolist(), heldout_subjects=[list(unit) for unit in sorted(held)],
        source_exposure=cfg['pareto']['front_exposure'], validation_examples_read=0))
    source_identity = dict(config=identity(cfg), input=identity(manifest),
        source_sha256=hashlib.sha256((ROOT / cfg['source']['model']).read_bytes()).hexdigest(),
        support_sha256=hashlib.sha256((ROOT / cfg['source']['support']).read_bytes()).hexdigest(),
        fold_sha256=hashlib.sha256((out / 'fold.json').read_bytes()).hexdigest())
    preserve_record(out / 'source_identity.json', source_identity)
    if (out / 'summary.json').exists():
        preserve_record(out/'completion_identity.json',dict(
            summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),
            fits={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.glob('k*_phase*/fit_*.npz'))}))
        return json.loads((out / 'summary.json').read_text())
    began = time.monotonic()
    cache, parts = OrderedDict(), []
    for offset in range(0, len(examples), 128):
        parts.append(base_tokenizer_batch(rows, examples[offset:offset+128], scales, array_cache=cache,
                                         cache_limit=cfg['resources']['mmap_parent_cache']))
    batch = {key: np.concatenate([part[key] for part in parts]) for key in ('x', 's', 'o', 'mask', 'vx', 'vs', 'vo')}
    del parts
    predicted = source.predict_continuous(batch['x'])
    train_keep = batch['mask'][~selection].all(-1).reshape(-1)
    held_keep = batch['mask'][selection].all(-1)
    sites = sorted({(rows[i]['dataset'], rows[i]['positions'][p]) for i, _, p in examples})
    site_index = {site: n for n, site in enumerate(sites)}
    locations = np.asarray([site_index[rows[i]['dataset'], rows[i]['positions'][p]] for i, _, p in examples])
    background = locations[:, None]*4 + np.searchsorted([.5, 1., 2.],
        np.sqrt(np.mean(batch['x']**2, axis=-1)), side='right')
    train_attr, held_attr = {}, {}
    for branch in ('S', 'O'):
        target = batch[branch.lower()]
        train_attr[branch] = np.exp(raw_features(target[~selection].reshape(-1,400)[train_keep],
            np.zeros(400), np.empty((0,400)), 'theta_alpha_rms'))
        held_attr[branch] = np.exp(raw_features(target[selection][held_keep],
            np.zeros(400), np.empty((0,400)), 'theta_alpha_rms'))
    cases = [(64, 0.)] + [(k, weight) for k in cfg['pareto']['codebook_sizes'] for weight in cfg['pareto']['waveform_weights']]
    results = []
    with threadpool_limits(limits=cfg['resources']['numerical_threads_per_worker']):
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as workers:
            for size, weight in cases:
                case = deepcopy(cfg)
                case['model'].update(codebook_size=size, waveform_weight=weight,
                    waveform_pca_dim=cfg['pareto']['waveform_pca_dim'] if weight else 0,
                    feature_mode='semantic_waveform' if weight else 'theta_alpha_rms')
                folder = out / f'k{size}_phase{weight:g}'
                folder.mkdir(exist_ok=True)
                preserve_record(folder / 'resolved_config.json', case)
                signature = dict(source_identity, case=identity(case))
                jobs = [workers.submit(fit_branch, branch, predicted[branch][~selection].reshape(-1,400)[train_keep],
                    batch[branch.lower()][~selection].reshape(-1,400)[train_keep], case, folder, signature) for branch in ('S', 'O')]
                fitted = dict(job.result() for job in as_completed(jobs))
                gains, codes = {}, {}
                fields = {name: {branch: fitted[branch][name] for branch in ('S', 'O')}
                          for name in ('means', 'bases', 'feature_mean', 'feature_scale', 'centers')}
                metadata = dict(schema='spectral_waveform_dictionary_v1', modality='eeg', codebook_size=size,
                    config_identity=identity(case), cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
                    input_identity=identity(manifest), source_sha256=source_identity['source_sha256'],
                    feature_mode=case['model']['feature_mode'], seed=cfg['training']['seed'], context_s=30, token_s=2,
                    diagnostic='training_subject_holdout_of_atom_fitting_conditioned_on_pretrained_front')
                model = SpectralWaveformDictionary(metadata, source.transfer,
                    atoms={branch:fitted[branch]['mean_atoms'] for branch in ('S','O')}, **fields)
                for branch in ('S','O'):
                    train_ids = model.nearest_ids(predicted[branch][~selection], branch).reshape(-1)[train_keep]
                    codes[branch] = model.nearest_ids(predicted[branch][selection], branch)
                    train_bg = background[~selection].reshape(-1)[train_keep]
                    held_bg = background[selection][held_keep]
                    totals = GroupTotals(len(sites)*4,3)
                    totals.add(train_bg,train_attr[branch])
                    pooled = train_attr[branch].mean(0)
                    train_base = np.where(totals.count[train_bg,None]>0,totals.mean(train_bg),pooled)
                    residuals = GroupTotals(size,3)
                    residuals.add(train_ids,train_attr[branch]-train_base)
                    held_base = np.where(totals.count[held_bg,None]>0,totals.mean(held_bg),pooled)
                    held_ids = codes[branch][held_keep]
                    shrink = residuals.count[held_ids]/(residuals.count[held_ids]+4.)
                    readout = held_base + shrink[:,None]*residuals.mean(held_ids)
                    error = np.zeros((*held_keep.shape,2,3))
                    error[held_keep] = np.stack(((held_base-held_attr[branch])**2,(readout-held_attr[branch])**2),axis=1)
                    average = error.sum(1)/held_keep.sum(1)[:,None,None].clip(1)
                    average[~held_keep.any(1)] = np.nan
                    aggregate = base_training_aggregate(dict(nmse=average),examples[selection].tolist(),rows)
                    values = np.asarray(aggregate['metrics']['nmse']['macro'])
                    gains[branch] = (1-values[1]/np.maximum(values[0],1e-30)).tolist()
                meaningful = all(g[0]>0 and g[1]>0 and max(g)>=.05 for g in gains.values())
                for mode in ('mean','energy'):
                    model.atoms = {branch:fitted[branch][mode+'_atoms'] for branch in ('S','O')}
                    prediction = model.decode_ids(codes)
                    held_batch = {key:value[selection] for key,value in batch.items()}
                    metrics = base_training_metrics(prediction,held_batch,1)
                    summary = base_training_aggregate(dict(nmse=metrics['nmse']),examples[selection].tolist(),rows)
                    record = dict(case=str(folder),K=size,phase_weight=weight,atom_mode=mode,
                        nmse=summary['metrics']['nmse'],score=summary['selection_score'],
                        ID_attribute_gain=gains,attribute_order=['theta_power','alpha_power','mean_local_RMS'],
                        component_semantic_gate=meaningful,heldout_subjects=len(held))
                    write_json(folder/(mode+'_cv.json'),record)
                    results.append(record)
                write_json(out/'progress.json',dict(completed_cases=len(results)//2,planned_cases=len(cases),
                    elapsed_seconds=time.monotonic()-began,validation_examples_read=0))
    eligible = [record for record in results if record['component_semantic_gate'] and record['phase_weight']>0]
    best = min(eligible,key=lambda record:record['score']) if eligible else None
    baseline = next(record for record in results if record['phase_weight']==0 and record['atom_mode']=='mean')
    summary = dict(status='training_component_CV_completed',results=results,best_semantics_preserving_case=best,
        baseline=baseline,seconds=time.monotonic()-began,validation_examples_read=0,confirm_examples_read=0,
        source_exposure=cfg['pareto']['front_exposure'],
        relative_NMSE_gain_vs_semantic_only=None if best is None else 1-best['score']/baseline['score'])
    write_json(out/'summary.json',summary)
    preserve_record(out/'completion_identity.json',dict(
        summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),
        fits={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.glob('k*_phase*/fit_*.npz'))}))
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_spectral_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codebook-size', type=int)
    parser.add_argument('--train-pareto', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.codebook_size is not None:
        cfg['model']['codebook_size'] = args.codebook_size
    with owner_lock(args.output.resolve()):
        try:
            if args.train_pareto:
                diagnose_pareto(cfg, args.output.resolve())
            else:
                run(cfg, args.output.resolve())
        except BaseException as error:
            if not (args.output / 'summary.json').exists():
                write_json(args.output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
