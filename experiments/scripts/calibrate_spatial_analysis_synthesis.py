#!/usr/bin/env python3
"""Fit fixed band gains on train AS integers and replay the same screen IDs."""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import yaml
from src.analysis.dual_component_tokenizer import eeg_complete_patch_band_energy
from src.analysis.spatial_tokenizer import screen, write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from src.tokenizers.spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, band_filter, load_model
from experiments.scripts.run_dual_component_tokenizer import identity, owner_lock
SELECTOR = np.array([[1.0] * 10, [1.0] * 5 + [0.0] * 5, [0.0] * 5 + [1.0] * 5])

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plan(rows, cfg):
    excluded = {tuple((c['row'], c['start_s'])) for c in json.loads((ROOT / cfg['source']['earlier_train_panel']).read_text())}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, row in enumerate(rows):
        if row['split'] == 'train' and row['modality'] == 'eeg':
            for item in row['inputs']['score']:
                if (i, item['start_s']) not in excluded:
                    groups[row['dataset']][row['subject']][row['record']].append(dict(row=i, start_s=item['start_s'], positions=item['positions']))

    def rank(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    result = []
    for dataset, subjects in sorted(groups.items()):
        chosen = sorted(subjects, key=lambda subject: rank(['AS-gain-v1', dataset, subject]))[:cfg['training']['subjects_per_dataset']]
        for subject in chosen:
            records = subjects[subject]
            for record in sorted(records, key=lambda record: rank([dataset, subject, record]))[:cfg['training']['records_per_subject']]:
                crop = min(records[record], key=lambda item: rank([dataset, subject, record, item['row'], item['start_s']]))
                result.append(crop)
    return result

def sufficient_statistics(features, targets, mask, variances):
    """Exact masked waveform and complete-patch Hann quadratic forms."""
    features = np.asarray(features, float)
    targets = np.asarray(targets, float)
    mask = np.asarray(mask, bool)
    if features.shape[1:] != (6000, 10) or targets.shape != (*features.shape[:2], 3) or mask.shape != features.shape[:2]:
        raise ValueError('full 30-second ten-feature/three-target layout required')
    if np.any(mask[..., None] & ~np.isfinite(targets)) or not np.isfinite(features).all():
        raise ValueError('nonfinite scored calibration values')
    count = mask.sum(1)
    if np.any(count == 0):
        raise ValueError('planned calibration item has no scoring support')
    safe = np.where(mask[..., None], targets, 0.0)
    weighted = features * mask[..., None]
    gram = np.einsum('nti,ntj->nij', weighted, features, optimize=True) / count[:, None, None]
    cross = np.einsum('nti,ntc->nci', weighted, safe, optimize=True) / count[:, None, None]
    energy = np.sum(safe ** 2, axis=1) / count[:, None]
    patch = features.reshape(len(features), 15, 400, 10)
    window = np.hanning(400)
    spectrum = np.fft.rfft((patch - patch.mean(2, keepdims=True)) * window[None, None, :, None], axis=2)
    full = mask.reshape(len(features), 15, 400).all(2)
    frequency = np.fft.rfftfreq(400, 1 / 200)
    power = []
    for lo, hi in [(4, 8), (8, 13)]:
        z = spectrum[:, :, (frequency >= lo) & (frequency < hi)]
        value = np.einsum('npfi,npfj->nij', z.conj() * full[:, :, None, None], z, optimize=True).real
        power.append(value * (2 / (400 * np.sum(window ** 2))) / full.sum(1)[:, None, None])
    target_power = []
    for c in range(3):
        values = eeg_complete_patch_band_energy(safe[:, :, c].reshape(-1, 15, 400, 1), mask.reshape(-1, 15, 400, 1))
        target_power.append(np.stack([values[4, 8][:, 0], values[8, 13][:, 0]], axis=1))
    return dict(gram=gram, cross=cross, energy=energy, variance=variances, power=np.stack(power, axis=1), target_power=np.stack(target_power, axis=1), valid_samples=count, full_patches=full.sum(1))

def item_metrics(data, gain):
    vectors = SELECTOR * gain[None]
    predicted = np.einsum('ci,nij,cj->nc', vectors, data['gram'], vectors, optimize=True)
    covariance = np.einsum('ci,nci->nc', vectors, data['cross'], optimize=True)
    nmse = (predicted - 2 * covariance + data['energy']) / data['variance']
    rms = np.sqrt(np.maximum(predicted, 0) / np.maximum(data['energy'], 1e-20))
    power = np.einsum('ci,nqij,cj->ncq', vectors, data['power'], vectors, optimize=True)
    db = 10 * np.log10(np.maximum(power, 1e-20) / np.maximum(data['target_power'], 1e-20))
    return dict(nmse=nmse, rms_gain=rms, band_db=db, band_abs_db=abs(db))

def weights(examples, rows):
    records = defaultdict(lambda: defaultdict(list))
    for j, (i, start, p) in enumerate(examples):
        row = rows[int(i)]
        records[row['subject']][row['record']].append(j)
    result = np.zeros(len(examples))
    for subject, group in records.items():
        for record, indices in group.items():
            crops = defaultdict(list)
            for j in indices:
                crops[tuple(examples[j][:2])].append(j)
            for crop, which in crops.items():
                result[which] = 1 / len(records) / len(group) / len(crops) / len(which)
    return result

def aggregate(data, gain, weight):
    return {key: np.tensordot(weight, value, axes=(0, 0)) for key, value in item_metrics(data, gain).items()}

def fit(data, weight, cfg, constrained):
    original = aggregate(data, np.ones(10), weight)
    limits = cfg['training']
    component_weight = np.array([0.5, 1.0, 1.0])
    normalized = weight[:, None] * component_weight[None] / data['variance']
    quadratic = np.einsum('nc,ci,nij,cj->ij', normalized, SELECTOR, data['gram'], SELECTOR, optimize=True)
    linear = np.einsum('nc,ci,nci->i', normalized, SELECTOR, data['cross'], optimize=True)
    constant = float(np.sum(normalized * data['energy']))

    def objective(gain):
        return float(gain @ quadratic @ gain - 2 * linear @ gain + constant)

    def gradient(gain):
        return 2 * (quadratic @ gain - linear)

    def restrictions(gain):
        m = aggregate(data, gain, weight)
        lo, hi = limits['constrained_rms_gain']
        db = limits['constrained_theta_alpha_mean_abs_db_max']
        return np.r_[m['rms_gain'] - lo, hi - m['rms_gain'], (db - abs(m['band_db'])).ravel(), (db - m['band_abs_db']).ravel(), original['nmse'] * limits['constrained_component_nmse_max_ratio_to_original_AS'] - m['nmse']]
    bounds = [tuple(limits['gain_bounds'])] * 10
    mse = minimize(objective, np.ones(10), method='L-BFGS-B', jac=gradient, bounds=bounds, options=dict(ftol=1e-12, maxiter=500))
    if not constrained:
        return dict(gain=mse.x, success=bool(mse.success), message=str(mse.message), objective=objective(mse.x), metrics=aggregate(data, mse.x, weight))
    moment = aggregate(data, np.ones(10), weight)
    initial = np.ones(10)
    for c, offset in [(1, 0), (2, 5)]:
        for band, q in [(1, 0), (2, 1)]:
            initial[offset + band] = np.clip(10 ** (-moment['band_db'][c, q] / 20), *limits['gain_bounds'])
    starts = [mse.x, initial]
    candidates = [minimize(objective, start, method='SLSQP', jac=gradient, bounds=bounds, constraints=[dict(type='ineq', fun=restrictions)], options=dict(ftol=1e-09, maxiter=500)) for start in starts]
    feasible = [v for v in candidates if np.min(restrictions(v.x)) >= -1e-06]
    selected = min(feasible, key=lambda v: objective(v.x)) if feasible else min(candidates, key=lambda v: float(np.sum(np.minimum(restrictions(v.x), 0) ** 2)))
    return dict(gain=selected.x, success=bool(selected.success), feasible=bool(feasible), message=str(selected.message), objective=objective(selected.x), minimum_constraint_margin=float(np.min(restrictions(selected.x))), metrics=aggregate(data, selected.x, weight), limits=limits)

def prepare(cfg, out, manifest, scales):
    rows = manifest['rows']
    panel = plan(rows, cfg)
    write(out / 'training_panel.json', panel)
    encoder = load_encoder(ROOT / cfg['source']['encoder_bundle'])
    decoder = load_model(ROOT / cfg['source']['decoder'])
    folder = out / 'train_statistics'
    folder.mkdir(exist_ok=True)
    cache = OrderedDict()
    paths = []
    for ordinal, crop in enumerate(panel):
        path = folder / f'crop_{ordinal:04d}.npz'
        paths.append(path)
        if path.exists():
            continue
        row = rows[crop['row']]
        dataset = row['dataset']
        names = decoder.metadata['positions'][dataset]
        fields, observed, _ = base_tokenizer_joint_eeg(rows, crop, scales, names, cache, allowed_splits=('train',))
        ids = encoder.encode(fields['x'].T.reshape(1, len(names), 15, 400), dataset, observed[None])
        prediction = decoder.decode_ids(ids, dataset)
        examples = [(crop['row'], crop['start_s'], p) for p in crop['positions']]
        batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=8)
        indices = [next((k for k, name in enumerate(names) if name.upper() == row['positions'][p].upper())) for p in crop['positions']]
        features = np.stack([band_filter(prediction[branch].reshape(len(names), 6000), band) for branch in ('S', 'O') for band in range(5)], axis=-1)[indices]
        targets = np.stack([batch[key].reshape(len(examples), 6000) for key in ('x', 's', 'o')], axis=-1)
        variances = np.stack([batch['v' + key][:, 0, 0] for key in ('x', 's', 'o')], axis=-1)
        statistics = sufficient_statistics(features, targets, batch['mask'].reshape(len(examples), 6000), variances)
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, examples=np.asarray(examples, np.int32), S=ids['S'], O=ids['O'], **statistics)
        temporary.replace(path)
        write(out / 'progress.json', dict(stage='train_statistics', completed=ordinal + 1, planned=len(panel)))
    write(out / 'training_search_trace.json', encoder.search_history)
    return paths

def load_statistics(paths):
    values = defaultdict(list)
    for path in paths:
        with np.load(path) as stored:
            for key in ('examples', 'gram', 'cross', 'energy', 'variance', 'power', 'target_power', 'valid_samples', 'full_patches'):
                values[key].append(stored[key])
    return {key: np.concatenate(items) for key, items in values.items()}

def fitting(cfg, out, data, rows):
    result = {arm: {} for arm in ('minimum_MSE', 'energy_constrained')}
    cv = []
    for dataset in sorted({rows[int(i)]['dataset'] for i, _, _ in data['examples']}):
        selected = np.array([rows[int(i)]['dataset'] == dataset for i, _, _ in data['examples']])
        local = {key: value[selected] for key, value in data.items()}
        weight = weights(local['examples'], rows)
        subjects = sorted({rows[int(i)]['subject'] for i, _, _ in local['examples']}, key=lambda s: hashlib.sha256(f'AS-gain-CV-v1/{dataset}/{s}'.encode()).hexdigest())
        for arm in result:
            result[arm][dataset] = fit(local, weight, cfg, arm == 'energy_constrained')
            for fold in range(4):
                held = {s for j, s in enumerate(subjects) if j % 4 == fold}
                keep = np.array([rows[int(i)]['subject'] not in held for i, _, _ in local['examples']])
                fit_data = {key: value[keep] for key, value in local.items()}
                held_data = {key: value[~keep] for key, value in local.items()}
                fitted = fit(fit_data, weights(fit_data['examples'], rows), cfg, arm == 'energy_constrained')
                cv.append(dict(dataset=dataset, arm=arm, fold=fold, held_subjects=sorted(held), fit_success=fitted['success'], fit_feasible=fitted.get('feasible'), gain=fitted['gain'], held_metrics=aggregate(held_data, fitted['gain'], weights(held_data['examples'], rows))))
            write(out / 'fit_progress.json', dict(completed_dataset=dataset, arm=arm))
    write(out / 'frozen_gains.json', dict(status='frozen_before_screen', source_encoder_sha256=digest(ROOT / cfg['source']['encoder_bundle']), source_decoder_sha256=digest(ROOT / cfg['source']['decoder']), arms=result, parameter_count=40, uses_cross_branch_mixing=False))
    write(out / 'training_subject_CV.json', cv)
    return result

def run(args, cfg):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with owner_lock(out):
        manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
        resolved = dict(config=cfg, config_identity=identity(cfg), input_identity=identity(manifest), scales_identity=manifest['scales_identity'], source_encoder_sha256=digest(ROOT / cfg['source']['encoder_bundle']), source_decoder_sha256=digest(ROOT / cfg['source']['decoder']))
        if (out / 'resolved_config.json').exists() and json.loads((out / 'resolved_config.json').read_text()) != resolved:
            raise ValueError('fixed calibration input identity changed')
        if (out / 'summary.json').exists():
            return
        write(out / 'resolved_config.json', resolved)
        checked = load_encoder(ROOT / cfg['source']['encoder_bundle'])
        decoder = load_model(ROOT / cfg['source']['decoder'])
        if checked.metadata['input_identity'] != manifest['cfg_identity'] or decoder.metadata['input_identity'] != manifest['cfg_identity'] or checked.metadata['scales_identity'] != manifest['scales_identity'] or (decoder.metadata['scales_identity'] != manifest['scales_identity']):
            raise ValueError('source encoder/decoder and G0 coordinate drift')
        paths = prepare(cfg, out, manifest, scales)
        data = load_statistics(paths)
        if (out / 'frozen_gains.json').exists():
            gains = json.loads((out / 'frozen_gains.json').read_text())['arms']
        else:
            gains = fitting(cfg, out, data, manifest['rows'])
        completed = []
        source = load_model(ROOT / cfg['source']['decoder'])
        for arm, group in gains.items():
            folder = out / arm
            folder.mkdir(exist_ok=True)
            if (folder / 'summary.json').exists():
                completed.append(json.loads((folder / 'summary.json').read_text()))
                continue
            readout = {}
            for dataset, record in group.items():
                gain = np.asarray(record['gain'])
                matrix = np.zeros((5, len(source.metadata['positions'][dataset]), 2, 2))
                for band in range(5):
                    matrix[band, :, 0, 0] = gain[band]
                    matrix[band, :, 1, 1] = gain[5 + band]
                readout[dataset] = matrix
            metadata = deepcopy(source.metadata)
            metadata.update(schema='spatial_waveform_dictionary_v2', calibration_arm=arm, calibration_config_identity=identity(cfg), encoder_bundle=str(ROOT / cfg['source']['encoder_bundle']), source_encoder_sha256=resolved['source_encoder_sha256'], free_calibration_parameters=40, serialized_readout_scalars=sum((value.size for value in readout.values())), readout_type='fixed_diagonal_band_gain_no_cross_branch_terms', calibration_train_subjects=len({(manifest['rows'][int(i)]['dataset'], manifest['rows'][int(i)]['subject']) for i, _, _ in data['examples']}))
            model = SpatialWaveformDictionary(metadata, source.transfer, source.centers, source.atoms, source.weights, readout)
            checkpoint = folder / 'model.npz'
            if not checkpoint.exists():
                model.save(checkpoint)
            summary = screen(model, checkpoint, manifest['rows'], scales, cfg, folder, ids_source=ROOT / cfg['source']['screen_ids'])
            record = dict(status='completed', arm=arm, checkpoint=str(checkpoint), screen_nmse=summary['spatial']['selection_score'], same_AS_integers=True, new_transmitted_bits=0, train_gain_fit_feasible=all((v.get('feasible', True) for v in group.values())))
            write(folder / 'summary.json', record)
            completed.append(record)
        write(out / 'summary.json', dict(status='completed', arms=completed, train_crops=len(paths), train_position_inputs=len(data['examples']), selection='whole calibrated arms frozen on train before both screen evaluations', test_arrays_read=0, confirmation_subjects_read=0))

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_as_gain_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['schema'] != 'dual_component_spatial_as_gain_v1' or cfg['training']['validation_statistics_fit'] != 'forbidden':
        raise ValueError('training-only fixed-gain contract required')
    with threadpool_limits(cfg['resources']['numerical_threads']):
        run(args, cfg)
if __name__ == '__main__':
    main()
