#!/usr/bin/env python3
"""Train-only 2x2 spectral readouts from two frozen spatial integer streams."""
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
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from src.analysis.spatial_tokenizer import screen, write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, band_filter, load_model, spectral_features

SOURCE_MODEL = None


def collect_dataset(task):
    dataset, subjects, rows, scales, cfg, output = task
    model = SOURCE_MODEL
    checkpoint = output / (dataset + '_statistics.npz')
    if checkpoint.exists():
        return str(checkpoint)
    names = model.metadata['positions'][dataset]
    roles = model.roles(dataset)
    maximum_sources = max(int(np.sum(roles == band)) for band in range(len(BANDS)))
    width = maximum_sources * 2
    cache = OrderedDict()
    result = []
    support = []
    integer_payload = {}
    integer_metadata = []
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        for subject, crops in subjects.items():
            gram = np.zeros((5, len(names), width, width))
            cross = np.zeros((5, len(names), width, 2))
            target_energy = np.zeros((5, len(names), 2))
            count = np.zeros(len(names))
            for crop in crops:
                fields, observed, valid = base_tokenizer_joint_eeg(rows, crop, scales, names, cache)
                support.append(dict(subject=subject, **crop, observed_positions=int(observed.sum()),
                                    complete_target_positions=int(valid.sum())))
                if not valid.any():
                    continue
                ids = model.encode(fields['x'].T.reshape(1, len(names), 15, 400), dataset, observed[None])
                ordinal = len(integer_metadata)
                integer_metadata.append(dict(crop=ordinal, dataset=dataset, subject=subject, **crop,
                    native_positions=names, complete_target_positions=np.flatnonzero(valid).tolist()))
                for branch in ('S', 'O'):
                    integer_payload[f'{branch}_{ordinal}'] = ids[branch]
                decoded = model.decode_ids(ids, dataset, spatial=False)
                count += valid
                for band in range(len(BANDS)):
                    anchors = np.flatnonzero(roles == band)
                    values = np.zeros((6000, width))
                    for column, branch in enumerate(('S', 'O')):
                        wave = decoded[branch][0].reshape(len(names), 6000).T
                        values[:, column * maximum_sources:column * maximum_sources + len(anchors)] = band_filter(wave, band, axis=0)[:, anchors]
                    target = np.stack([band_filter(fields[field], band, axis=0) for field in ('s', 'o')], axis=-1)
                    gram[band, valid] += values.T @ values / 6000
                    cross[band, valid] += np.einsum('tm,tpo->pmo', values, target[:, valid], optimize=True) / 6000
                    target_energy[band, valid] += np.mean(target[:, valid]**2, axis=0)
            gram /= np.maximum(count, 1)[None, :, None, None]
            cross /= np.maximum(count, 1)[None, :, None, None]
            target_energy /= np.maximum(count, 1)[None, :, None]
            result.append(dict(subject=subject, gram=gram, cross=cross, target_energy=target_energy, valid=count > 0))
            write(output / (dataset + '_progress.json'), dict(stage='training_integer_statistics',
                completed_subjects=len(result), planned_subjects=len(subjects)))
    np.savez_compressed(checkpoint, subjects=np.array([item['subject'] for item in result]),
                        gram=np.stack([item['gram'] for item in result]), cross=np.stack([item['cross'] for item in result]),
                        target_energy=np.stack([item['target_energy'] for item in result]), valid=np.stack([item['valid'] for item in result]))
    np.savez_compressed(output / (dataset + '_training_ids.npz'), **integer_payload)
    write(output / (dataset + '_training_integer_metadata.json'), integer_metadata)
    write(output / (dataset + '_support.json'), support)
    return str(checkpoint)


def average_statistics(gram, cross, target, valid):
    denominator = np.maximum(valid.sum(axis=0), 1)
    return (gram.sum(axis=0) / denominator[None, :, None, None],
            cross.sum(axis=0) / denominator[None, :, None, None],
            target.sum(axis=0) / denominator[None, :, None])


def reduced_moments(gram, cross, spatial):
    transpose = spatial.swapaxes(-1, -2)
    return transpose @ gram @ spatial, transpose @ cross


def minimum_mse(gram, cross, cutoff):
    return np.linalg.pinv(gram, rcond=cutoff, hermitian=True) @ cross


def diagonal_minimum_mse(gram, cross, cutoff):
    diagonal = np.diagonal(gram, axis1=-2, axis2=-1)
    threshold = cutoff * np.max(diagonal, axis=-1, keepdims=True)
    supported = diagonal > threshold
    result = np.zeros_like(cross)
    indices = np.arange(2)
    result[..., indices, indices] = np.divide(
        np.diagonal(cross, axis1=-2, axis2=-1), diagonal,
        out=np.zeros_like(diagonal), where=supported)
    return result


def spatial_from_statistics(model, dataset, gram, cross, fraction):
    roles = model.roles(dataset)
    maximum_sources = gram.shape[-1] // 2
    result = np.zeros((*gram.shape[:3], 2))
    for band in range(len(BANDS)):
        count = int(np.sum(roles == band))
        for column in range(2):
            indices = np.arange(count) + column * maximum_sources
            g = gram[band][:, indices][:, :, indices].copy()
            c = cross[band][:, indices, column]
            regularization = fraction * np.trace(g, axis1=1, axis2=2) / count + 1e-12
            diagonal = np.arange(count)
            g[:, diagonal, diagonal] += regularization[:, None]
            result[band][:, indices, column] = np.linalg.solve(g, c[..., None])[..., 0]
    return result


def frozen_spatial(model, dataset, shape):
    roles = model.roles(dataset)
    maximum_sources = shape[-1] // 2
    result = np.zeros((*shape[:3], 2))
    for band in range(len(BANDS)):
        count = int(np.sum(roles == band))
        for column, branch in enumerate(('S', 'O')):
            result[band, :, column * maximum_sources:column * maximum_sources + count, column] = model.weights[dataset, branch, band].T
    return result


def fit_readouts(model, dataset, statistics, cfg, *, diagonal=False):
    subjects = statistics['subjects']
    gram, cross, target, valid = (statistics[key] for key in ('gram', 'cross', 'target_energy', 'valid'))
    full_gram, full_cross, full_target = average_statistics(gram, cross, target, valid)
    spatial = frozen_spatial(model, dataset, full_gram.shape)
    reduced_gram, reduced_cross = reduced_moments(full_gram, full_cross, spatial)
    cutoff = cfg['readout']['pseudo_inverse_relative_cutoff']
    fit = diagonal_minimum_mse if diagonal else minimum_mse
    coefficients = fit(reduced_gram, reduced_cross, cutoff)
    ordered = sorted(range(len(subjects)), key=lambda i: hashlib.sha256((dataset + str(subjects[i])).encode()).hexdigest())
    folds = np.zeros(len(subjects), int)
    for rank, index in enumerate(ordered):
        folds[index] = rank % cfg['readout']['crossfit_folds']
    crossfit_prediction = np.zeros_like(target)
    crossfit_covariance = np.zeros_like(target)
    crossfit_mse = np.zeros_like(target)
    for fold in range(cfg['readout']['crossfit_folds']):
        fitting = folds != fold
        held = np.flatnonzero(folds == fold)
        g, c, t = average_statistics(gram[fitting], cross[fitting], target[fitting], valid[fitting])
        fold_spatial = spatial_from_statistics(model, dataset, g, c, cfg['readout']['source_spatial_ridge_fraction'])
        local_gram, local_cross = reduced_moments(g, c, fold_spatial)
        fold_coefficients = fit(local_gram, local_cross, cutoff)
        for index in held:
            held_gram, held_cross = reduced_moments(gram[index], cross[index], fold_spatial)
            prediction = np.diagonal(fold_coefficients.swapaxes(-1, -2) @ held_gram @ fold_coefficients, axis1=-2, axis2=-1)
            covariance = np.diagonal(fold_coefficients.swapaxes(-1, -2) @ held_cross, axis1=-2, axis2=-1)
            crossfit_prediction[index] = prediction
            crossfit_covariance[index] = covariance
            crossfit_mse[index] = target[index] - 2 * covariance + prediction
    denominator = np.maximum(valid.sum(axis=0), 1)[None, :, None]
    predicted = crossfit_prediction.sum(axis=0) / denominator
    covariance = crossfit_covariance.sum(axis=0) / denominator
    correlation = covariance / np.sqrt(np.maximum(predicted * full_target, 1e-20))
    requested_gain = np.sqrt(full_target / np.maximum(predicted, 1e-20))
    setting = cfg['readout']['energy_gain']
    eligible = (correlation >= setting['required_crossfit_correlation']) & (predicted > 1e-10) & (full_target > 1e-10)
    gain = np.where(eligible, np.clip(requested_gain, setting['minimum_gain'], setting['maximum_gain']), setting['unsupported_gain'])
    energy_coefficients = coefficients * gain[..., None, :]
    summary = dict(dataset=dataset, subjects=subjects.tolist(), subject_folds=folds.tolist(),
        coefficient_axes=['frequency_band', 'native_target_position', 'input_S_O', 'output_S_O'],
        minimum_mse_coefficients=coefficients, energy_coefficients=energy_coefficients,
        crossfit_correlation=correlation, crossfit_prediction_energy=predicted, target_training_energy=full_target,
        requested_energy_gain=requested_gain, applied_energy_gain=gain, energy_eligible=eligible,
        crossfit_mean_mse=crossfit_mse.sum(axis=0) / denominator,
        crossfit_scope='spatial W and 2x2 readout refitted without the held training subjects; encoder and dictionaries remain frozen train models',
        input='same-position voltages decoded from the entire native-layout S and O integer sequences',
        readout_constraint='diagonal: each output depends only on its own branch' if diagonal else 'full 2x2')
    return coefficients, energy_coefficients, summary


def evaluate_task(task):
    variant, condition, cfg, rows, scales, output = task
    checkpoint = output / (variant + '.npz')
    destination = output / variant / condition
    destination.mkdir(exist_ok=True, parents=True)
    if (destination / 'screen_summary.json').exists():
        return str(destination / 'screen_summary.json')
    model = load_model(checkpoint)
    options = {'cross_branch': False} if condition == 'same_branch' else {}
    intervention = condition if condition not in ('full', 'same_branch') else None
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        screen(model, checkpoint, rows, scales, cfg, destination, ROOT / cfg['source']['integers'],
               decoder_options=options, intervention=intervention, mode_names=('spatial',), render=condition == 'full')
    return str(destination / 'screen_summary.json')


def synthetic(output):
    rng = np.random.default_rng(61)
    inputs = rng.normal(size=(10000, 3, 2))
    truth_coefficients = rng.normal(size=(3, 2, 2))
    targets = np.einsum('tpi,pio->tpo', inputs, truth_coefficients)
    gram = np.einsum('tpi,tpj->pij', inputs, inputs) / len(inputs)
    cross = np.einsum('tpi,tpo->pio', inputs, targets) / len(inputs)
    fitted = minimum_mse(gram, cross, 1e-6)
    np.testing.assert_allclose(fitted, truth_coefficients, atol=1e-12)
    diagonal = diagonal_minimum_mse(gram, cross, 1e-6)
    indices = np.arange(2)
    np.testing.assert_allclose(diagonal[..., indices, indices],
        np.diagonal(cross, axis1=-2, axis2=-1) / np.diagonal(gram, axis1=-2, axis2=-1), atol=1e-12)
    np.testing.assert_array_equal(diagonal[..., 0, 1], 0)
    np.testing.assert_array_equal(diagonal[..., 1, 0], 0)
    metadata = dict(schema='spatial_waveform_dictionary_v2', codebook_size=4, positions={'fixture': list('ABCDE')}, output_projection_hz=[1, 45])
    centers = {(branch, band): rng.normal(size=(3, spectral_features(np.zeros((1, 400)), band).shape[-1]))
               for branch in ('S', 'O') for band in range(5)}
    atoms = {(branch, band): np.r_[np.zeros((1, 400)), rng.normal(size=(3, 400))]
             for branch in ('S', 'O') for band in range(5)}
    weights = {(dataset, branch, band): np.eye(5)[band:band + 1] for dataset in ('fixture',) for branch in ('S', 'O') for band in range(5)}
    readout = {'fixture': np.tile(np.eye(2), (5, 5, 1, 1))}
    model = SpatialWaveformDictionary(metadata, {branch: np.full(3001, .5) for branch in ('S', 'O')}, centers, atoms, weights, readout)
    ids = {branch: rng.integers(1, 4, size=(1, 5, 15), dtype=np.int32) for branch in ('S', 'O')}
    raw = model.decode_ids(ids, 'fixture', cross_branch=False)
    output_values = model.decode_ids(ids, 'fixture')
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_allclose(raw[branch], output_values[branch], atol=1e-12)
    model.save(output / 'synthetic_model.npz')
    restored = load_model(output / 'synthetic_model.npz')
    for branch, value in output_values.items():
        np.testing.assert_array_equal(value, restored.decode_ids(ids, 'fixture')[branch])
    write(output / 'synthetic.json', dict(status='passed', known_2x2_coefficients_recovered=True,
        identity_readout_preserves_output=True, integer_only_checkpoint_replay='exact', decoder_continuous_side_information=0))


def render_matched(cfg, rows, scales, output, *, worst=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = {'same-branch spatial': load_model(ROOT / cfg['source']['model']),
              'minimum-MSE 2x2': load_model(output / 'minimum_mse.npz'),
              'train-energy 2x2': load_model(output / 'train_energy.npz')}
    metadata = json.loads((ROOT / cfg['source']['integers']).with_name('integer_metadata.json').read_text())
    with np.load(ROOT / cfg['evaluation']['panel']) as stored:
        examples = stored['eeg_screen']
    selected = {}
    for item in metadata:
        selected.setdefault(item['dataset'], item)
    worst_indices = {}
    if worst:
        with np.load(output / 'train_energy/full/spatial_metrics.npz') as stored:
            scores = np.mean(stored['nmse'], axis=(1, 2))
        for dataset in selected:
            indices = [index for index, (row_index, _, _) in enumerate(examples) if rows[row_index]['dataset'] == dataset]
            worst_indices[dataset] = max(indices, key=lambda index: scores[index] if np.isfinite(scores[index]) else float('inf'))
            row_index, start, _ = examples[worst_indices[dataset]]
            selected[dataset] = next(item for item in metadata if item['row'] == row_index and item['start_s'] == start)
    identities = []
    cache = OrderedDict()
    with np.load(ROOT / cfg['source']['integers']) as integers:
        for dataset, item in selected.items():
            row_index, start = item['row'], item['start_s']
            row = rows[row_index]
            if row['split'] != 'validation':
                raise ValueError('fixed screen visual support required')
            panel_index = worst_indices[dataset] if worst else int(np.flatnonzero((examples[:, 0] == row_index) & (examples[:, 1] == start))[0])
            native_position = int(examples[panel_index, 2])
            name = row['positions'][native_position]
            names = models['same-branch spatial'].metadata['positions'][dataset]
            position = next(i for i, value in enumerate(names) if value.upper() == name.upper())
            ids = {branch: integers[f"{branch}_{item['crop']}"] for branch in ('S', 'O')}
            predictions = {label: model.decode_ids(ids, dataset) for label, model in models.items()}
            batch = base_tokenizer_batch(rows, [(row_index, start, native_position)], scales, array_cache=cache, cache_limit=8)
            scale = scales['positions']['eeg'][dataset][name]['scale']
            time_axis = np.arange(6000) / 200
            fig, axes = plt.subplots(3, 3, figsize=(16, 8), constrained_layout=True)
            styles = [('same-branch spatial', '#3977ad', '--'), ('minimum-MSE 2x2', '#d28220', '-.'),
                      ('train-energy 2x2', '#21834a', '-')]
            for axis_row, (branch, field) in enumerate((('S', 's'), ('O', 'o'), ('sum', 'x'))):
                truth = batch[field][0].reshape(-1) * scale
                curves = [(label, values[branch][0, position].reshape(-1) * scale) for label, values in predictions.items()]
                for column, bounds in enumerate(((0, 30), (14, 16), (13.5, 14.5))):
                    ax = axes[axis_row, column]
                    ax.plot(time_axis, truth, color='#252525', linewidth=.9, label='target')
                    for label, color, linestyle in styles:
                        values = next(value for key, value in curves if key == label)
                        ax.plot(time_axis, values, color=color, linestyle=linestyle, linewidth=.85, label=label)
                    visible = (time_axis >= bounds[0]) & (time_axis < bounds[1])
                    values = [truth] + [value for _, value in curves]
                    low = min(float(np.nanmin(value[visible])) for value in values)
                    high = max(float(np.nanmax(value[visible])) for value in values)
                    margin = .05 * max(high - low, 1e-20)
                    ax.set_ylim(low - margin, high + margin)
                    ax.set_xlim(*bounds)
                    for boundary in np.arange(2, 30, 2):
                        ax.axvline(boundary, color='grey', linewidth=.5, linestyle=':')
                    invalid = ~batch['mask'][0].reshape(-1)
                    transitions = np.diff(np.r_[False, invalid, False].astype(int))
                    for begin, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                        ax.axvspan(begin / 200, end / 200, color='grey', alpha=.25)
                    ax.set_xlabel('Window time (s)')
                    ax.set_ylabel(('X=S+O' if branch == 'sum' else branch) + ' (' + row['source_unit'] + ')')
                    if axis_row == 0 and column == 0:
                        ax.legend(fontsize=8)
            fig.suptitle(f"Same saved integers / {dataset} / {row['subject']} / {row['record']} / {name} / crop {start} s\nDotted lines mark 2-second token boundaries; grey shading marks unscored samples")
            suffix = '_matched_worst_waveform.png' if worst else '_matched_waveform.png'
            path = output / (dataset + suffix)
            fig.savefig(path, dpi=200)
            plt.close(fig)
            identities.append(dict(dataset=dataset, subject=row['subject'], record=row['record'], parent=row['parent'],
                manifest_row=row_index, native_start_sample=row['native_start_sample'], native_rate_hz=row['native_rate_hz'],
                crop_start_s=start, position=name, panel_index=panel_index, figure=str(path),
                selection='candidate worst mean XSO G0 NMSE per dataset' if worst else
                          'first registered joint screen crop per dataset and first selected position in that crop'))
    write(output / ('matched_worst_visual_examples.json' if worst else 'matched_visual_examples.json'), identities)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_cross_branch_readout_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--visual-selection', choices=('fixed', 'worst'), default='fixed')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.synthetic:
        synthetic(args.output)
        return
    cfg = yaml.safe_load(args.config.read_text())
    diagonal = cfg['schema'] == 'dual_component_cross_branch_readout_control_v1'
    variants = ['single_branch_minimum_mse', 'single_branch_train_energy'] if diagonal else ['minimum_mse', 'train_energy']
    if (cfg['schema'] not in ('dual_component_cross_branch_readout_v1', 'dual_component_cross_branch_readout_control_v1')
            or cfg['readout']['fitting_split'] != 'train' or cfg['readout']['variants'] != variants):
        raise ValueError('versioned train-only frozen-codebook readout required')
    if (args.output / 'summary.json').exists() and not args.render_only:
        raise FileExistsError('retained completed readout experiment')
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    global SOURCE_MODEL
    SOURCE_MODEL = load_model(ROOT / cfg['source']['model'])
    if (SOURCE_MODEL.metadata['input_identity'] != manifest['cfg_identity']
            or SOURCE_MODEL.metadata['scales_identity'] != manifest['scales_identity']):
        raise ValueError('frozen source G0 identity changed')
    if args.render_only:
        render_matched(cfg, rows, scales, args.output, worst=args.visual_selection == 'worst')
        return
    plan = json.loads((ROOT / cfg['source']['training_plan']).read_text())['plan']
    if any(rows[crop['row']]['split'] != 'train' for subjects in plan.values() for crops in subjects.values() for crop in crops):
        raise ValueError('train readout boundary before arrays')
    resolved = dict(config=cfg, source_model_sha256=hashlib.sha256((ROOT / cfg['source']['model']).read_bytes()).hexdigest(),
                    input_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'], training_plan=plan)
    statistics_root = ROOT / cfg['source']['training_statistics'] if diagonal else args.output
    if diagonal:
        source_resolved = json.loads((statistics_root / 'resolved_config.json').read_text())
        for key in ('source_model_sha256', 'input_identity', 'scales_identity', 'training_plan'):
            if source_resolved[key] != resolved[key]:
                raise ValueError('matched single-branch training statistics changed: ' + key)
        resolved['training_statistics_sha256'] = {dataset: hashlib.sha256((statistics_root / (dataset + '_statistics.npz')).read_bytes()).hexdigest()
            for dataset in sorted(plan)}
    path = args.output / 'resolved_config.json'
    if path.exists() and json.loads(path.read_text()) != resolved:
        raise ValueError('readout run configuration changed')
    write(path, resolved)
    started = time.monotonic()
    if not diagonal:
        tasks = [(dataset, subjects, rows, scales, cfg, args.output) for dataset, subjects in sorted(plan.items())]
        with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers'], mp_context=multiprocessing.get_context('fork')) as pool:
            list(pool.map(collect_dataset, tasks))
    fitted = {variant: {} for variant in variants}
    diagnostics = []
    for dataset in sorted(plan):
        with np.load(statistics_root / (dataset + '_statistics.npz')) as stored:
            minimum, energy, summary = fit_readouts(SOURCE_MODEL, dataset, {key: stored[key] for key in stored.files}, cfg, diagonal=diagonal)
        fitted[variants[0]][dataset] = minimum
        fitted[variants[1]][dataset] = energy
        diagnostics.append(summary)
    write(args.output / 'training_readout.json', diagnostics)
    original_metadata = json.loads(json.dumps(SOURCE_MODEL.metadata))
    for variant, matrices in fitted.items():
        SOURCE_MODEL.metadata = dict(original_metadata, schema='spatial_waveform_dictionary_v2', readout_variant=variant,
            cross_branch_readout='frozen_per_dataset_native_position_band_diagonal_linear_no_intercept' if diagonal else
                'frozen_per_dataset_native_position_band_2x2_linear_no_intercept',
            readout_fit='train_only', readout_source_sha256=resolved['source_model_sha256'],
            source_model=str(ROOT / cfg['source']['model']), decoder_branch_dependencies=
                {'S': ['S_IDs'], 'O': ['O_IDs'], 'sum': ['S_IDs', 'O_IDs']} if diagonal else ['S_IDs', 'O_IDs'])
        SOURCE_MODEL.readout = matrices
        destination = args.output / (variant + '.npz')
        if destination.exists():
            retained = load_model(destination)
            if retained.metadata != SOURCE_MODEL.metadata:
                raise ValueError('retained readout checkpoint identity changed')
            for dataset, value in matrices.items():
                np.testing.assert_array_equal(retained.readout[dataset], value)
        else:
            SOURCE_MODEL.save(destination)
    evaluations = [(variant, 'full', cfg, rows, scales, args.output) for variant in fitted]
    evaluations.extend(('minimum_mse', condition, cfg, rows, scales, args.output) for condition in cfg['evaluation']['interventions'])
    with ProcessPoolExecutor(max_workers=cfg['resources']['screen_workers'], mp_context=multiprocessing.get_context('fork')) as pool:
        summaries = list(pool.map(evaluate_task, evaluations))
    scores = {}
    for path in summaries:
        path = Path(path)
        summary = json.loads(path.read_text())['summaries']['spatial']
        scores[path.parent.parent.name + '/' + path.parent.name] = summary['metrics']['nmse']['macro']
    write(args.output / 'summary.json', dict(status='completed', nmse_component_order=['X', 'S', 'O'],
        paired_screen_NMSE=scores, elapsed_seconds=time.monotonic() - started,
        fitting='train_only; source encoders/codebooks frozen; spatial and readout crossfit on five training-subject folds for energy calibration',
        integer_budget='identical complete native-layout sequences as projection_v1, including all auxiliary positions',
        interpretation='cross-branch decoder diagnostic; separate single-branch local-ID audit remains required'))


if __name__ == '__main__':
    main()
