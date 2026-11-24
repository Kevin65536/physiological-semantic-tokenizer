"""Training-subject-selected shrinkage diagnostics for retained single-ID audits."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit_joint_tokenizer_patterns import pattern_attributes, hierarchy, serial, write_json
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_batch


LAMBDAS = [np.inf, 1024., 256., 64., 16., 4., 0.]


def background_prediction(audit, ids, background):
    mean = audit.local.total.sum(0) / max(int(audit.local.count.sum()), 1)
    return np.where(audit.background.count[background, None] > 0, audit.background.mean(background), mean)


def additive_residual(audit):
    """Training ID means of residuals after the fitted categorical background."""
    totals = audit.local.total.copy()
    for joint, count in audit.joint.count.items():
        bg, code = divmod(joint, audit.k)
        totals[code] -= count * audit.background.mean(np.asarray([bg]))[0]
    return totals / audit.local.count[:, None].clip(1)


def readout_predictions(audit, ids, background, residual):
    base = background_prediction(audit, ids, background)
    joint_keys = background * audit.k + ids
    joint = audit.joint.lookup(joint_keys, base)
    joint_count = np.asarray([audit.joint.count.get(int(key), 0) for key in joint_keys])
    result = [base]
    for method in ('joint', 'additive'):
        count = joint_count if method == 'joint' else audit.local.count[ids]
        increment = joint - base if method == 'joint' else residual[ids]
        for prior in LAMBDAS:
            weight = np.zeros(len(ids)) if np.isinf(prior) else count / np.maximum(count + prior, 1)
            result.append(base + weight[:, None] * increment)
    return result, joint_count


def subject_partition(rows, examples):
    """Metadata-only fifth of training subjects per dataset, fixed before readout scores."""
    subjects = {}
    for i, _, _ in examples:
        row = rows[i]
        subjects.setdefault(row['dataset'], set()).add(row['subject'])
    held = set()
    for dataset, values in subjects.items():
        ranked = sorted(values, key=lambda value: hashlib.sha256(
            f'joint-readout-variance-v1/{dataset}/{value}'.encode()).hexdigest())
        held.update((dataset, value) for value in ranked[:max(1, len(ranked) // 5)])
    select = np.asarray([(rows[i]['dataset'], rows[i]['subject']) in held for i, _, _ in examples])
    if select.all() or not select.any():
        raise ValueError('training subjects must support separate readout fitting and selection')
    return select, sorted(held)


def batch_attributes(rows, examples, scales, modality, site_index, cache):
    batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=128)
    valid = batch['mask'].all(-1)
    location = np.asarray([site_index.get((rows[i]['dataset'], rows[i]['positions'][p]), len(site_index))
                           for i, _, p in examples])
    rms = np.sqrt(np.mean(batch['x'] ** 2, axis=-1))
    background = location[:, None] * 4 + np.searchsorted([.5, 1., 2.], rms, side='right')
    target = {branch: pattern_attributes(batch[branch.lower()][valid], modality)[0] for branch in 'SO'}
    return valid, background[valid], target


def score(audits, residuals, rows, examples, ids, scales, modality, site_index, cache, batch_size):
    fields = audits['S'].fields
    errors = {b: [] for b in 'SO'}
    supports = {b: [] for b in 'SO'}
    for start in range(0, len(examples), batch_size):
        take = examples[start:start + batch_size]
        valid, background, targets = batch_attributes(rows, take, scales, modality, site_index, cache)
        for branch, audit in audits.items():
            code = ids[branch][start:start + len(take)][valid]
            predictions, counts = readout_predictions(audit, code, background, residuals[branch])
            patch_errors = np.stack([feature_means((p - targets[branch]) ** 2, fields) for p in predictions], axis=1)
            values = np.zeros((*valid.shape, len(predictions), len(fields)))
            values[valid] = patch_errors
            average = values.sum(1) / valid.sum(1)[:, None, None].clip(1)
            average[~valid.any(1)] = np.nan
            errors[branch].append(average)
            supports[branch].append(counts)
    return {b: np.concatenate(value) for b, value in errors.items()}, {b: np.concatenate(v) for b, v in supports.items()}


def load_context(source, panel_role):
    from run_dual_component_tokenizer import identity, read_config
    from src.data.dual_component_tokenizer import base_tokenizer_training_data
    launch = json.loads((source/'launch.json').read_text())
    command = launch['command']
    def option(name, default=None):
        return command[command.index(name) + 1] if name in command else default
    checkpoint = Path(option('--checkpoint'))
    modality = option('--modality')
    kind = option('--kind', 'dictionary')
    panels_path = Path(option('--panels'))
    config = read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml')
    manifest, scales, _, _ = base_tokenizer_training_data(config, ROOT)
    panel_record = json.loads((panels_path.parent/'panel.json').read_text())
    if (panel_record['cfg_identity'] != manifest['cfg_identity']
            or panel_record['scales_identity'] != manifest['scales_identity']
            or panel_record['panel_sha256'] != hashlib.sha256(panels_path.read_bytes()).hexdigest()):
        raise ValueError('frozen panel identity before signal arrays')
    if kind == 'legacy':
        import torch
        legacy_run = Path(option('--legacy-run'))
        legacy_config = json.loads((legacy_run/'resolved_config.json').read_text())['config']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        signature = saved['signature']
        task = dict(modality=modality, architecture='waveform', candidate='vq128')
        if (signature['config'] != identity(legacy_config) or signature['task'] != task
                or signature['input'] != identity(manifest) or signature['scales'] != identity(scales)):
            raise ValueError('legacy checkpoint identity before signal arrays')
        metadata = dict(schema='legacy_joint_pattern_adapter_v1', modality=modality,
                        codebook_size=legacy_config['model']['codebook_size'],
                        cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
                        seed=legacy_config['training']['seed'], source_signature=signature)
    elif kind == 'contextual':
        from src.tokenizers.contextual_latent_dictionary import load_model
    elif kind == 'spectral':
        from src.tokenizers.spectral_waveform_dictionary import load_model
    elif kind == 'fullband':
        from src.tokenizers.fullband_eeg_tokenizer import load_model
    else:
        from src.tokenizers.supervised_waveform_dictionary import load_model
    if kind != 'legacy':
        metadata = load_model(checkpoint).metadata
    if metadata['modality'] != modality:
        raise ValueError('checkpoint modality before signal arrays')
    if kind == 'dictionary':
        if metadata['input_identity'] != identity(manifest):
            raise ValueError('checkpoint G0 identity before signal arrays')
    elif (metadata['cfg_identity'] != manifest['cfg_identity']
          or metadata['scales_identity'] != manifest['scales_identity']):
        raise ValueError('checkpoint G0 identity before signal arrays')
    with np.load(panels_path) as panel:
        training = panel[modality + '_pattern_train'].copy()
        validation = panel[modality + '_' + panel_role].copy()
    codes = {}
    for split, examples in [('train', training), ('validation', validation)]:
        with np.load(source/(split + '_IDs.npz')) as archive:
            np.testing.assert_array_equal(archive['examples'], examples)
            codes[split] = {b: archive[b].copy() for b in 'SO'}
        expected = 'train' if split == 'train' else 'validation'
        if any(manifest['rows'][i]['split'] != expected or manifest['rows'][i]['modality'] != modality
               for i, _, _ in examples):
            raise ValueError('readout split/modality before signal arrays')
        for code in codes[split].values():
            if (code.shape != (len(examples), 15) or not np.issubdtype(code.dtype, np.integer)
                    or code.min(initial=0) < 0 or code.max(initial=0) >= metadata['codebook_size']):
                raise ValueError('retained IDs outside frozen interface')
    return manifest['rows'], scales, training, validation, codes, metadata, panel_record, modality


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, scales, train, validation, codes, metadata, panel_record, modality = load_context(args.source, args.panel_role)
    held, held_subjects = subject_partition(rows, train)
    sites = sorted({(r['dataset'], p) for r in rows if r['modality'] == modality and r['split'] == 'train'
                    for p in r['positions']})
    site_index = {site: n for n, site in enumerate(sites)}
    datasets = sorted({r['dataset'] for r in rows if r['modality'] == modality})
    width = 400 if modality == 'eeg' else 40
    probe, fields = pattern_attributes(np.zeros((1, width)), modality)
    size = metadata['codebook_size']
    make_audits = lambda: {b: SparsePatternAudit(size, (len(sites) + 1) * 4, datasets, probe.shape[-1], fields) for b in 'SO'}
    full, fitting = make_audits(), make_audits()
    cache = OrderedDict()
    began = time.monotonic()
    for start in range(0, len(train), args.batch_size):
        take = train[start:start + args.batch_size]
        valid, background, targets = batch_attributes(rows, take, scales, modality, site_index, cache)
        fit_patch = np.broadcast_to(~held[start:start + len(take), None], valid.shape)[valid]
        for b in 'SO':
            code = codes['train'][b][start:start + len(take)][valid]
            full[b].fit(code, background, targets[b])
            fitting[b].fit(code[fit_patch], background[fit_patch], targets[b][fit_patch])
        write_json(out/'progress.json', dict(stage='train_sufficient_statistics', inputs=start + len(take), planned=len(train)))
    cv_examples = train[held]
    cv_ids = {b: codes['train'][b][held] for b in 'SO'}
    residuals = {b: additive_residual(audit) for b, audit in fitting.items()}
    cv_errors, _ = score(fitting, residuals, rows, cv_examples, cv_ids, scales, modality, site_index, cache, args.batch_size)
    primary = ['power_4_8', 'power_8_13', 'rms_envelope'] if modality == 'eeg' else [
        c + s for c in ('HbO', 'HbR') for s in ('_slope', '_centered')]
    primary_index = [list(fields).index(name) for name in primary]
    selected = {}
    cv_summaries = {}
    for b in 'SO':
        cv_summaries[b] = hierarchy(cv_errors[b], cv_examples, rows)
        macro = np.asarray(cv_summaries[b]['macro'])
        objective = np.nanmean(macro[:, primary_index] / np.maximum(macro[0, primary_index], 1e-30), axis=1)
        selected[b] = {method: 1 + index * len(LAMBDAS) + int(np.argmin(objective[1 + index * len(LAMBDAS):1 + (index + 1) * len(LAMBDAS)]))
                       for index, method in enumerate(('joint', 'additive'))}
    del fitting
    residuals = {b: additive_residual(audit) for b, audit in full.items()}
    val_errors, val_support = score(full, residuals, rows, validation, codes['validation'], scales,
                                    modality, site_index, cache, args.batch_size)
    source_summary_path = args.source/'numerical_summary.json'
    if not source_summary_path.exists():
        source_summary_path = args.source/'summary.json'
    old = json.loads(source_summary_path.read_text())
    labels = ['background'] + [method + '_lambda_' + ('infinity' if np.isinf(prior) else str(int(prior)))
                               for method in ('joint', 'additive') for prior in LAMBDAS]
    summary = dict(status='completed', source=str(args.source.resolve()), checkpoint_metadata=metadata,
                   panel_role=args.panel_role, panel_sha256=panel_record['panel_sha256'], attributes=list(fields),
                   primary_attributes=primary, candidate_order=labels,
                   selection='one fixed metadata-hashed fifth of training subjects per dataset held out from readout fitting',
                   objective='equal mean primary-attribute MSE divided by training-holdout background MSE; ties favor stronger shrinkage',
                   held_training_subjects=held_subjects, training_contexts=len(train), selection_contexts=int(held.sum()),
                   validation_contexts=len(validation), elapsed_seconds=time.monotonic() - began, branches={})
    for b in 'SO':
        scores = hierarchy(val_errors[b], validation, rows)
        macro = np.asarray(scores['macro'])
        original = np.asarray(old['branches'][b]['attribute_mse']['macro'])
        np.testing.assert_allclose(macro[0], original[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(macro[1 + len(LAMBDAS) - 1], original[2], rtol=1e-10, atol=1e-12)
        counts = val_support[b]
        training_counts = np.asarray(list(full[b].joint.count.values()))
        gains = {method: dict(zip(fields, 1 - macro[index] / macro[0])) for method, index in selected[b].items()}
        summary['branches'][b] = dict(selected={method: labels[index] for method, index in selected[b].items()},
            gain_over_background=gains, unsmoothed_joint_gain=dict(zip(fields, 1 - original[2] / original[0])),
            original_saturated_scores_reproduced=True, training_holdout_scores=cv_summaries[b], screen_scores=scores,
            occupied_training_joint_cells=len(training_counts),
            training_joint_cell_count_quantiles=np.quantile(training_counts, [0, .1, .25, .5, .9, 1]),
            validation_joint_training_count_quantiles=np.quantile(counts, [0, .1, .25, .5, .9, 1]),
            validation_joint_mass_unseen=float(np.mean(counts == 0)),
            validation_joint_mass_below_8=float(np.mean(counts < 8)))
    np.savez_compressed(out/'training_holdout_errors.npz', examples=cv_examples, **cv_errors)
    np.savez_compressed(out/'panel_errors.npz', examples=validation, **val_errors)
    write_json(out/'summary.json', summary)
    write_json(out/'progress.json', dict(stage='completed', elapsed_seconds=time.monotonic() - began))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--panel-role', choices=['screen', 'development', 'confirm'], default='screen')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
