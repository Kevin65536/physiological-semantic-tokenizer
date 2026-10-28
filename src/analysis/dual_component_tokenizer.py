"""Train-only controls, subject-level evidence, and frozen-token mask pilot.

The cache is already divided by observation-training SD without centering. MSE
in this space is NMSE against that common frozen denominator, for S/O/sum alike.
Windows and seeds are never treated as independent biological replicates.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import hashlib
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from src.foundation.dual_component_masked import (
    DualComponentMaskedModel, block_masks, own_context, expected_waveforms,
    fit_token_baselines, baseline_probabilities, nonwrapping_source_shift,
)

MODALITIES = ('eeg', 'hb')
BRANCHES = ('S', 'O')


def _json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=convert, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def _cache(run):
    cache = Path(run) / 'cache'
    manifest = json.loads((cache / 'manifest.json').read_text())
    arrays = {k: np.load(cache / manifest.get('arrays', {}).get(k, k + '.npy'), mmap_mode='r')
              for m in MODALITIES for k in (m + '_u', m + '_s', m + '_e')}
    if any(len(x) != len(manifest['rows']) for x in arrays.values()):
        raise ValueError('cache row identity/array shape mismatch')
    if any(not np.allclose(arrays[m + '_s'] + arrays[m + '_e'], arrays[m + '_u'], atol=2e-5)
           for m in MODALITIES):
        raise ValueError('cached teacher decomposition identity failed')
    return cache, manifest, arrays


def _region(row):
    return str(row.get('roi', row.get('region', row.get('anchor', {}).get('region', ''))))


def normalized_roughness(x, channels=1):
    """Per-channel mean squared 100 ms difference / variance; patch-local."""
    x = np.asarray(x, float).reshape(*np.asarray(x).shape[:-1], -1, channels)
    axes = tuple(range(x.ndim - 1))
    return np.mean(np.diff(x, axis=-2) ** 2, axis=axes) / np.maximum(np.var(x, axis=axes), 1e-12)


def fit_matched_smoothing(observed, physical, training, sigmas=(0., 1., 2., 4., 8.), *, channels=1):
    """Choose each channel's smoothing width using training roughness alone."""
    observed, physical = np.asarray(observed), np.asarray(physical)
    training = np.asarray(training, bool)
    if observed.shape != physical.shape or len(training) != len(observed) or not training.any():
        raise ValueError('same-support observations/teacher and training mask required')
    sigmas = tuple(float(s) for s in sigmas)
    if not sigmas or min(sigmas) < 0:
        raise ValueError('nonnegative candidate Gaussian widths required')
    desired = normalized_roughness(physical[training], channels)
    reshaped = observed.reshape(*observed.shape[:-1], -1, channels)
    candidates = [reshaped if s == 0 else gaussian_filter1d(reshaped, s, axis=-2, mode='nearest')
                  for s in sigmas]
    roughness = np.stack([normalized_roughness(x[training].reshape(observed[training].shape), channels)
                          for x in candidates])
    mismatch = np.abs(np.log(np.maximum(roughness, 1e-12)) - np.log(np.maximum(desired, 1e-12)))
    selected = np.argmin(mismatch, axis=0)
    result = np.stack([candidates[selected[c]][..., c] for c in range(channels)], axis=-1)
    report = dict(sigmas_samples=[sigmas[j] for j in selected],
        sigmas_seconds=[.1 * sigmas[j] for j in selected], candidate_sigmas_samples=sigmas,
        teacher_training_roughness=desired, selected_training_roughness=roughness[selected, np.arange(channels)],
        selection='train-only per-channel absolute log normalized-roughness mismatch',
        support='within each 2-second patch; nearest endpoint; no cross-patch smoothing',
        fit_rows=int(training.sum()), eval_used_for_selection=False)
    return result.reshape(observed.shape).astype('float32'), report


def fit_smooth_targets(cfg, cache_manifest, cache_dir):
    """Return a manifest for fixed matched-smoothing S and its full residual."""
    cache = Path(cache_dir); rows = cache_manifest['rows']
    train = np.array([r['split'] == 'train' for r in rows])
    smoothing = cfg.get('smoothing', cfg.get('analysis', {}).get('smoothing', {}))
    sigmas = smoothing.get('sigma_steps', smoothing.get('sigmas_samples', smoothing.get('sigmas', [0., 1., 2., 4., 8.])))
    result = dict(schema='dual_component_matched_smoothing_v1', arrays={}, modalities={})
    for modality in MODALITIES:
        def read(part):
            key = modality + '_' + part
            return np.load(cache / cache_manifest.get('arrays', {}).get(key, key + '.npy'), mmap_mode='r')
        u, s = read('u'), read('s')
        smooth, report = fit_matched_smoothing(u, s, train, sigmas, channels=2 if modality == 'hb' else 1)
        for name, data in (('s', smooth), ('e', np.asarray(u) - smooth)):
            filename = 'smooth_' + modality + '_' + name + '.npy'
            path = cache / filename
            if path.exists():
                existing = np.load(path, mmap_mode='r')
                if existing.shape != data.shape or not np.array_equal(existing, data):
                    raise ValueError('retained smoothing target changed; use a fresh run namespace')
            else:
                np.save(path, data.astype('float32'))
            result['arrays'][modality + '_' + name] = filename
        result['modalities'][modality] = report
    _json(cache / 'smoothing.json', result)
    return result


def reconstruction_rows(predictions, physical, remaining, observed, rows, *, variant, seed, target_family='Tak_teacher'):
    """Window scores; all three denominators are observation-training SD squared."""
    result = []
    for m in MODALITIES:
        targets = (dict(sum=observed[m]) if variant == 'ordinary_hard'
                   else dict(S=physical[m], O=remaining[m], sum=observed[m]))
        for branch, target in targets.items():
            value = np.asarray(predictions[m + '_' + branch], float)
            target = np.asarray(target, float)
            if value.shape != target.shape or len(value) != len(rows):
                raise ValueError('predictions/targets/identities must share complete support')
            selectors = [('EEG_proxy' if m == 'eeg' else 'Hb_joint', slice(None))]
            if m == 'hb': selectors.extend([('HbO', slice(0, None, 2)), ('HbR', slice(1, None, 2))])
            for signal, selector in selectors:
                difference = value[..., selector] - target[..., selector]
                finite = np.isfinite(difference).all(axis=tuple(range(1, value.ndim)))
                errors = np.mean(difference ** 2, axis=tuple(range(1, value.ndim)))
                for i, row in enumerate(rows):
                    result.append(dict(dataset=row.get('dataset', ''), subject=row['subject'],
                        record=row.get('record', row.get('join_key', '')), split=row['split'],
                        variant=variant, seed=int(seed), modality=m, component=branch, signal=signal,
                        target_family='processed_observation' if branch == 'sum' else target_family,
                        NMSE=float(errors[i]) if finite[i] else None,
                        windows=1, failures=int(not finite[i]),
                        branch_semantics=('arbitrary_additive_streams' if variant == 'ordinary_hard'
                                          else 'matched_smoothing' if variant == 'smooth_hard' else 'Tak_teacher')))
    return result


def subject_aggregate(rows, *, value='NMSE'):
    """Windows -> equal records -> subject; retain failures and observed counts."""
    excluded = {'subject', 'record', value, 'windows', 'failures'}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple((k, str(v)) for k, v in row.items() if k not in excluded)
        groups[(key, str(row['subject']))][str(row.get('record', ''))].append(row)
    result = []
    for (key, subject), records in groups.items():
        means = []
        total, failures = 0, 0
        for record_rows in records.values():
            values = [float(r[value]) for r in record_rows if r.get(value) is not None and np.isfinite(r[value])]
            total += len(record_rows)
            failures += sum(int(r.get('failures', r.get(value) is None)) for r in record_rows)
            if values: means.append(float(np.mean(values)))
        result.append(dict(key, subject=subject, **{value:float(np.mean(means)) if means else None},
            records=len(records), windows=total, failures=failures))
    return result


def paired_subject_bootstrap(first, second, *, value='NMSE', samples=2000, seed=0):
    """Average seeds within subjects, then paired subject bootstrap; first-second."""
    def reduce(rows):
        values = defaultdict(list)
        for row in rows:
            x = row.get(value)
            if x is not None and np.isfinite(x):
                values[(str(row.get('dataset', '')), str(row['subject']))].append(float(x))
        return {key:float(np.mean(x)) for key, x in values.items()}
    a, b = reduce(first), reduce(second)
    common = sorted(a.keys() & b.keys())
    difference = np.array([a[s] - b[s] for s in common])
    result = dict(subjects=len(common), first_subjects=len(a), second_subjects=len(b),
        unmatched_first=len(a.keys() - b.keys()), unmatched_second=len(b.keys() - a.keys()),
        effect='first minus second; lower NMSE favors first',
        mean_difference=float(difference.mean()) if len(difference) else None,
        interval_type='percentile paired-subject bootstrap, seeds averaged within subject',
        bootstrap_samples=int(samples), CI95=None,
        inference='exploratory; no multiplicity correction; windows/seeds are not independent subjects')
    if len(common) >= 2 and samples >= 1:
        rng = np.random.default_rng(seed)
        draws = difference[rng.integers(len(common), size=(samples, len(common)))].mean(axis=1)
        result['CI95'] = np.quantile(draws, [.025, .975]).tolist()
    return result


def codebook_health(ids, prototypes, rows, *, modality, branch, variant, seed, observed=None):
    """Split-specific occupancy and independent subject/record coverage per code."""
    ids, prototypes = np.asarray(ids), np.asarray(prototypes)
    if ids.ndim != 2 or prototypes.ndim != 2 or len(ids) != len(rows):
        raise ValueError('codebook health requires [window,time] IDs and [code,waveform] prototypes')
    if ids.dtype.kind not in 'iu': raise ValueError('hard integer IDs required')
    result = []
    for split in ('train', 'validation', 'test'):
        selected = np.array([r['split'] == split for r in rows])
        split_ids = ids[selected]
        valid = split_ids[(split_ids >= 0) & (split_ids < len(prototypes))]
        counts = np.bincount(valid, minlength=len(prototypes))
        p = counts / max(int(counts.sum()), 1)
        entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        observed_rms = np.sqrt(np.mean(np.asarray(observed, float) ** 2, axis=-1)) if observed is not None else None
        for k, prototype in enumerate(prototypes):
            assignment_mask = selected[:, None] & (ids == k)
            amplitudes = observed_rms[assignment_mask] if observed_rms is not None else np.array([])
            assigned = np.flatnonzero(selected & np.any(ids == k, axis=1))
            result.append(dict(variant=variant, seed=seed, modality=modality, component=branch,
                split=split, code=k, assignments=int(counts[k]), probability=float(p[k]),
                assigned_windows=len(assigned), subjects=len({(rows[i].get('dataset', ''), rows[i]['subject']) for i in assigned}),
                records=len({(rows[i].get('dataset', ''), rows[i].get('record', rows[i].get('join_key', ''))) for i in assigned}),
                assigned_observation_RMS_P10=float(np.quantile(amplitudes, .1)) if len(amplitudes) else None,
                assigned_observation_RMS_P50=float(np.quantile(amplitudes, .5)) if len(amplitudes) else None,
                assigned_observation_RMS_P90=float(np.quantile(amplitudes, .9)) if len(amplitudes) else None,
                prototype_mean=float(prototype.mean()), prototype_RMS=float(np.sqrt(np.mean(prototype ** 2))),
                prototype_min=float(prototype.min()), prototype_max=float(prototype.max()),
                active_codes=int((counts > 0).sum()), perplexity=float(np.exp(entropy)),
                dead_fraction=float(np.mean(counts == 0))))
    return result


def align_codewords(reference, candidate):
    """Training-frozen waveform correspondence, not equality of arbitrary IDs."""
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('same code count and decoded waveform dimensions required')
    cost = np.mean((a[:, None] - b[None]) ** 2, axis=-1)
    i, j = linear_sum_assignment(cost)
    mapping = np.empty(len(b), dtype=int); mapping[j] = i
    return mapping, dict(mean_prototype_NMSE=float(cost[i, j].mean()),
                        P90_prototype_NMSE=float(np.quantile(cost[i, j], .9)))


def _job(cfg, run, variant, seed):
    return Path(run) / 'jobs' / variant / ('seed_' + str(seed))


def _subject_plot(path, rows):
    if not rows: return
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    selected = [r for r in rows if r['split'] == 'test' and r['component'] == 'sum' and r['NMSE'] is not None and r.get('signal') in ('EEG_proxy', 'Hb_joint')]
    variants = list(dict.fromkeys(r['variant'] for r in selected))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, m in zip(axes, MODALITIES):
        for j, variant in enumerate(variants):
            values = [r['NMSE'] for r in selected if r['variant'] == variant and r['modality'] == m]
            if values: ax.scatter(np.full(len(values), j), values, s=14, alpha=.45)
        ax.set(xticks=np.arange(len(variants)), xticklabels=variants,
               ylabel='Observation MSE / frozen training SD²', title=m + ': subject-equal record means')
        ax.tick_params(axis='x', labelrotation=35); ax.grid(alpha=.2)
    fig.savefig(path, dpi=180); plt.close(fig)


def analyze_suite(cfg, run):
    """Read only this run's explicit completed job exports; write shared tables."""
    run = Path(run); cache, manifest, arrays = _cache(run); rows = manifest['rows']
    settings = cfg.get('analysis', {})
    variants = cfg.get('variants', cfg.get('training', {}).get('variants', ['continuous_student', 'hard_student',
                'component_compression', 'smooth_hard', 'ordinary_hard']))
    seeds = cfg.get('seeds', cfg.get('training', {}).get('seeds', [17, 29, 43]))
    physical = {m:arrays[m + '_s'] for m in MODALITIES}
    remaining = {m:arrays[m + '_e'] for m in MODALITIES}
    observed = {m:arrays[m + '_u'] for m in MODALITIES}
    subjects, health, alignment, missing = [], [], [], []
    reference = {}
    for variant in variants:
        for seed in seeds:
            job = _job(cfg, run, variant, seed)
            record = json.loads((job / 'summary.json').read_text()) if (job / 'summary.json').exists() else {}
            if (record.get('status') != 'completed' or record.get('step') != int(cfg.get('training', {}).get('steps', 3000))
                    or not (job / 'predictions.npz').exists()):
                missing.append(dict(variant=variant, seed=seed, reason='completed_fixed_budget_job_and_prediction_export_required')); continue
            with np.load(job / 'predictions.npz', allow_pickle=False) as data:
                predictions = {key:data[key] for key in data.files}
            component_physical, component_remaining = physical, remaining
            target_family = 'Tak_teacher'
            if variant == 'smooth_hard':
                smoothing = json.loads((cache / 'smoothing.json').read_text())
                component_physical = {m:np.load(cache / smoothing['arrays'][m + '_s'], mmap_mode='r') for m in MODALITIES}
                component_remaining = {m:np.load(cache / smoothing['arrays'][m + '_e'], mmap_mode='r') for m in MODALITIES}
                if any(not np.allclose(component_physical[m] + component_remaining[m], observed[m], atol=2e-5) for m in MODALITIES):
                    raise ValueError('matched smoothing targets must sum to the same processed observation')
                target_family = 'matched_smoothing'
            windows = reconstruction_rows(predictions, component_physical, component_remaining, observed, rows,
                                           variant=variant, seed=seed, target_family=target_family)
            subjects.extend(subject_aggregate(windows))
            if (job / 'codewords.npz').exists():
                with np.load(job / 'codewords.npz', allow_pickle=False) as data:
                    codewords = {key:data[key] for key in data.files}
                for m in MODALITIES:
                    for branch in BRANCHES:
                        key = m + '_' + branch; ids_key = m + '_ids_' + branch
                        if key not in codewords or ids_key not in predictions: continue
                        health.extend(codebook_health(predictions[ids_key], codewords[key], rows,
                            modality=m, branch=branch, variant=variant, seed=seed, observed=observed[m]))
                        rkey = (variant, m, branch)
                        if rkey not in reference:
                            reference[rkey] = (seed, codewords[key], predictions[ids_key])
                        else:
                            ref_seed, ref_book, ref_ids = reference[rkey]
                            mapping, effect = align_codewords(ref_book, codewords[key])
                            current = predictions[ids_key]; valid = (current >= 0) & (ref_ids >= 0)
                            aligned = mapping[np.maximum(current, 0)]
                            for split in ('train', 'validation', 'test'):
                                selected = np.array([r['split'] == split for r in rows])[:, None] & valid
                                alignment.append(dict(variant=variant, modality=m, component=branch,
                                    reference_seed=ref_seed, seed=seed, split=split, **effect,
                                    aligned_agreement=float(np.mean(aligned[selected] == ref_ids[selected])) if selected.any() else None,
                                    matching='Hungarian distance of fixed decoded waveforms; no eval fitting'))
    # Simple fixed-output controls are fit on train only and evaluated everywhere.
    train = np.array([r['split'] == 'train' for r in rows])
    if train.any():
        for baseline in ('training_mean', 'zero'):
            predictions = {}
            for m in MODALITIES:
                for branch, values in (('S', physical[m]), ('O', remaining[m])):
                    center = np.mean(values[train], axis=(0, 1), keepdims=True) if baseline == 'training_mean' else np.zeros((1, 1, values.shape[-1]))
                    predictions[m + '_' + branch] = np.broadcast_to(center, values.shape)
                predictions[m + '_sum'] = predictions[m + '_S'] + predictions[m + '_O']
            subjects.extend(subject_aggregate(reconstruction_rows(predictions, physical, remaining, observed,
                rows, variant=baseline, seed=0)))
    comparisons = []
    for dataset in sorted({r.get('dataset', '') for r in rows}):
        for split in ('validation', 'test'):
            for m, signal in (('eeg', 'EEG_proxy'), ('hb', 'Hb_joint'), ('hb', 'HbO'), ('hb', 'HbR')):
                for branch in ('S', 'O', 'sum'):
                    def pick(variant):
                        return [r for r in subjects if r['variant'] == variant and r['split'] == split
                                and r['modality'] == m and r['component'] == branch and r['signal'] == signal
                                and r.get('dataset', '') == dataset]
                    for control in ('continuous_student', 'component_compression', 'smooth_hard', 'ordinary_hard', 'training_mean'):
                        if control in ('smooth_hard', 'ordinary_hard') and branch != 'sum': continue
                        effect = paired_subject_bootstrap(pick('hard_student'), pick(control),
                            samples=int(settings.get('bootstrap_samples', cfg.get('evaluation', {}).get('bootstrap_repeats', 1000))), seed=int(cfg.get('seed', 0)))
                        comparisons.append(dict(first='hard_student', second=control, dataset=dataset, split=split,
                            modality=m, component=branch, signal=signal,
                            target_family='processed_observation' if branch == 'sum' else 'Tak_teacher', **effect))
    output = run / 'analysis'; output.mkdir(exist_ok=True)
    _csv(output / 'reconstruction_subjects.csv', subjects)
    _csv(output / 'codebook_health.csv', health)
    _csv(output / 'seed_alignment.csv', alignment)
    _csv(output / 'paired_comparisons.csv', comparisons)
    _subject_plot(output / 'reconstruction_subjects.png', subjects)
    sensitivity = analyze_teacher_sensitivity(cfg, run)
    quality = vocabulary_quality_screen(subjects, cfg)
    masked = summarize_masked_pilots(cfg, run)
    summary = dict(schema='dual_component_analysis_v1', status='completed' if not missing else 'partial',
        missing_jobs=missing, reconstruction_subject_rows=len(subjects), codebook_rows=len(health),
        paired_comparisons=comparisons, teacher_sensitivity=sensitivity, quality_screen=quality, masked=masked,
        physical_compatibility=dict(status='not_evaluated',
            reason='No Tak-compatible whole-trajectory witness was evaluated for decoded masked predictions; 20-second patches retain the 90-second teacher reference coordinate'),
        planned_windows=manifest.get('planned_windows', len(rows)), retained_windows=len(rows),
        failed_trials=manifest.get('failed_trials', []),
        metric='mean squared error in shared frozen observation-training-SD coordinates; no per-window scaling',
        aggregation='equal windows within record; equal records within subject; seeds averaged before paired subject bootstrap',
        scope='processed-observation offline completion; full-record Hb preprocessing retained',
        cross_vocabulary_CE_ranking_allowed=False)
    _json(output / 'summary.json', summary)
    return summary


def _checkpoint(path, model, optimizer, step, rng, losses, signature):
    payload = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        step=step, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        losses=losses, signature=signature, cuda_rng=torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None)
    tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)


def _fit_reader(ids, rows, config, output, *, device, seed, reader):
    """Fixed-exposure train-only fitting, resumable independently per reader."""
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = DualComponentMaskedModel({m:int(config['vocab_size']) for m in MODALITIES},
        patches=ids.shape[1], width=int(config.get('width', 128)), layers=int(config.get('layers', 2)),
        heads=int(config.get('heads', 4)), dropout=float(config.get('dropout', .1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 3e-4)),
                                  weight_decay=float(config.get('weight_decay', 1e-4)))
    rng = np.random.default_rng(seed); start = 0; losses = []
    path = output / 'checkpoint.pt'
    signature = dict(reader=reader, seed=seed, config=config, id_shape=ids.shape,
        training_identities=[r.get('id', str(i)) for i, r in enumerate(rows) if r['split'] == 'train'])
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get('signature') != signature or saved['step'] > int(config.get('steps', 1000)):
            raise ValueError('masked checkpoint identity/config/reader/budget mismatch')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        start = int(saved['step']); rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu()); losses = saved.get('losses', [])
        if saved.get('cuda_rng') is not None and str(device).startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
    datasets = sorted({r.get('dataset', '') for r in rows if r['split'] == 'train'})
    groups = [np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and r.get('dataset', '') == d]) for d in datasets]
    if not groups or any(not len(g) for g in groups): raise ValueError('train data missing')
    steps = int(config.get('steps', 1000)); batch_size = int(config.get('batch_size', 64))
    block = int(config.get('block_patches', 2)); checkpoint_every = int(config.get('checkpoint_every', 250))
    began = time.monotonic(); model.train()
    for step in range(start, steps):
        selected = rng.choice(groups[int(rng.integers(len(groups)))], batch_size, replace=True)
        batch = ids[selected]; target_modality = int(rng.integers(len(MODALITIES)))
        task = 'time' if rng.integers(2) == 0 else 'modality'
        visible, target = block_masks(batch >= 0, rng=rng, target_modality=target_modality,
                                      block_patches=block, task=task)
        if reader == 'own': visible = own_context(visible, target_modality)
        truth = torch.as_tensor(batch, dtype=torch.long, device=device)
        target = torch.as_tensor(target, device=device)
        logits = model(truth, torch.as_tensor(visible, device=device))
        branch_losses = []
        for c in range(2):
            keep = target[..., c]
            if keep.any(): branch_losses.append(F.cross_entropy(logits[..., c, :][keep], truth[..., c][keep]))
        if not branch_losses: raise ValueError('empty masked target support')
        loss = torch.stack(branch_losses).sum()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('gradient_clip', 5.)))
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            losses.append(dict(step=step + 1, loss=float(loss.detach()), task=task, target_modality=MODALITIES[target_modality]))
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            _checkpoint(path, model, optimizer, step + 1, rng, losses, signature)
            _json(output / 'progress.json', dict(step=step + 1, steps=steps, reader=reader,
                last_loss=float(loss.detach()), resumed_from=start, elapsed_s=time.monotonic() - began))
    _csv(output / 'training_loss.csv', losses)
    model.eval()
    return model


def _predict(model, ids, visible, *, device, batch_size=128):
    result = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            logits = model(torch.as_tensor(ids[start:start + batch_size], device=device, dtype=torch.long),
                           torch.as_tensor(visible[start:start + batch_size], device=device))
            result.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(result)


def matched_donor_tokens(ids, rows, decoded, *, source_modality, training_ids,
                         training_rows, seed=0):
    """Same dataset/ROI/time, training-amplitude-bin donor, different subject.

    Matching uses only source token IDs and fixed decoded vocabulary, never the
    target, teacher, task/condition labels or eval-fitted amplitude thresholds.
    Donors come from the same evaluation partition and preserve patch index.
    """
    rng = np.random.default_rng(seed); m = int(source_modality)
    def amplitude(x):
        waves = sum(decoded[(m, c)][np.maximum(x[:, :, m, c], 0)] for c in range(2))
        return np.sqrt(np.mean(waves ** 2, axis=-1))
    training_amp = amplitude(training_ids)
    cutoffs = np.quantile(training_amp, [1 / 3, 2 / 3])
    amplitude_bins = np.searchsorted(cutoffs, amplitude(ids))
    result = ids.copy(); available = np.zeros(ids.shape[:2], bool)
    pools = defaultdict(list)
    for j, row in enumerate(rows):
        for t in range(ids.shape[1]):
            if np.all(ids[j, t, m] >= 0):
                pools[(str(row.get('dataset', '')), str(row.get('split', '')), _region(row), t, int(amplitude_bins[j, t]))].append(j)
    for i, row in enumerate(rows):
        for t in range(ids.shape[1]):
            key = (str(row.get('dataset', '')), str(row.get('split', '')), _region(row), t, int(amplitude_bins[i, t]))
            pool = pools.get(key, [])
            donor = None
            # Usually succeeds immediately, while remaining bounded and exact.
            for _ in range(min(8, len(pool))):
                candidate = pool[int(rng.integers(len(pool)))]
                if rows[candidate]['subject'] != row['subject']:
                    donor = candidate; break
            if donor is None:
                candidates = [j for j in pool if rows[j]['subject'] != row['subject']]
                if candidates: donor = int(rng.choice(candidates))
            if donor is not None:
                result[i, t, m] = ids[donor, t, m]
                available[i, t] = True
    return result, available, dict(amplitude_cutoffs_training_SD=cutoffs,
        matching='dataset, ROI, patch phase and train-fixed source decoded-amplitude tertile; other subject',
        eligible_source_positions=int(available.sum()), planned_source_positions=int(available.size),
        missing_donor='source tokens hidden; target scores retained', task_labels_used=False)


def _metric_subjects(window_rows):
    """One row per subject and metric, preserving trial/record dependence."""
    return subject_aggregate(window_rows, value='value')


def _score_probabilities(probability, ids, target, rows, decoded, arrays, marginal, metadata,
                         *, route, seed, task, context, target_modality):
    m = target_modality; modality = MODALITIES[m]; result = []
    predicted = {c:expected_waveforms(probability[:, :, m, c], decoded[(m, c)]) for c in range(2)}
    oracle = {c:decoded[(m, c)][np.maximum(ids[:, :, m, c], 0)] for c in range(2)}
    branch_targets = {0:arrays[modality + '_s'], 1:arrays[modality + '_e']}
    def add(values, selected, component, metric):
        for i, row in enumerate(rows):
            keep = selected[i]
            if not keep.any(): continue
            value = values[i][keep]
            result.append(dict(dataset=row.get('dataset', ''), subject=row['subject'],
                record=row.get('record', row.get('join_key', '')), split=row['split'],
                variant=route, seed=seed, task=task, context=context, modality=modality,
                component=component, metric=metric,
                target_family=('processed_observation' if component == 'sum' else 'arbitrary_additive_streams' if route == 'ordinary_hard' else 'matched_smoothing' if route == 'smooth_hard' else 'Tak_teacher'),
                value=float(np.mean(value)) if np.isfinite(value).all() else None,
                failures=int(not np.isfinite(value).all())))
    for c, branch in enumerate(BRANCHES):
        selected = target[:, :, m, c]
        truth = np.maximum(ids[:, :, m, c], 0)[..., None]
        def ce(p): return -np.log(np.maximum(np.take_along_axis(p[:, :, m, c], truth, axis=-1)[..., 0], 1e-12))
        error, marginal_error, metadata_error = ce(probability), ce(marginal), ce(metadata)
        add(error, selected, branch, 'CE_nats')
        entropy = -np.sum(probability[:, :, m, c] * np.log(np.maximum(probability[:, :, m, c], 1e-12)), axis=-1)
        add(entropy, selected, branch, 'predictive_entropy_nats')
        add(marginal_error - error, selected, branch, 'DeltaCE_marginal_nats')
        add(metadata_error - error, selected, branch, 'DeltaCE_metadata_nats')
        add(np.mean((predicted[c] - oracle[c]) ** 2, axis=-1), selected, branch, 'model_to_tokenizer_NMSE')
        if route != 'ordinary_hard':
            add(np.mean((predicted[c] - branch_targets[c]) ** 2, axis=-1), selected, branch, 'NMSE')
            add(np.mean((oracle[c] - branch_targets[c]) ** 2, axis=-1), selected, branch, 'tokenizer_NMSE')
            if modality == 'hb':
                for channel, name in enumerate(('HbO', 'HbR')):
                    add(np.mean((predicted[c][..., channel::2] - branch_targets[c][..., channel::2]) ** 2, axis=-1), selected, branch, 'NMSE_' + name)
    selected = target[:, :, m].all(-1)
    summed = predicted[0] + predicted[1]; oracle_sum = oracle[0] + oracle[1]
    add(np.mean((summed - arrays[modality + '_u']) ** 2, axis=-1), selected, 'sum', 'NMSE')
    if modality == 'hb':
        for channel, name in enumerate(('HbO', 'HbR')):
            add(np.mean((summed[..., channel::2] - arrays[modality + '_u'][..., channel::2]) ** 2, axis=-1), selected, 'sum', 'NMSE_' + name)
    if route != 'ordinary_hard':
        channels = 2 if modality == 'hb' else 1
        assembled = np.where(target[:, :, m, 0, None], predicted[0], oracle[0])
        segments = assembled.reshape(*assembled.shape[:2], -1, channels)
        teacher_segments = branch_targets[0].reshape(segments.shape)
        signed_jumps = segments[:, 1:, 0] - segments[:, :-1, -1]
        teacher_signed_jumps = teacher_segments[:, 1:, 0] - teacher_segments[:, :-1, -1]
        jumps = np.mean(signed_jumps ** 2, axis=-1)
        true_jumps = np.mean(teacher_signed_jumps ** 2, axis=-1)
        jump_error = np.mean((signed_jumps - teacher_signed_jumps) ** 2, axis=-1)
        boundaries = target[:, 1:, m, 0] | target[:, :-1, m, 0]
        add(np.pad(jumps, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_squared_trainingSD')
        add(np.pad(jumps - true_jumps, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_excess_over_teacher_trainingSD')
        add(np.pad(jump_error, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_error_MSE_trainingSD2')
    add(np.mean((oracle_sum - arrays[modality + '_u']) ** 2, axis=-1), selected, 'sum', 'tokenizer_NMSE')
    add(np.mean((summed - oracle_sum) ** 2, axis=-1), selected, 'sum', 'model_to_tokenizer_NMSE')
    return result, predicted


def _trace_plot(path, rows, arrays, oracle, predicted, target, modality, route):
    if not len(rows): return
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    channels = 2 if modality == 'hb' else 1
    shape = (len(rows), -1, channels)
    fig, axes = plt.subplots(3, channels, figsize=(11, 7), squeeze=False, constrained_layout=True)
    time_axis = np.arange(np.asarray(arrays[modality + '_u'])[0].size // channels) * .1
    s = np.asarray(arrays[modality + '_s']).reshape(shape)
    o = np.asarray(arrays[modality + '_e']).reshape(shape)
    u = np.asarray(arrays[modality + '_u']).reshape(shape)
    mask = np.repeat(target[0], arrays[modality + '_u'].shape[-1] // channels)
    for c in range(channels):
        for a, (teacher, dec, pred, title) in enumerate(((s, oracle[0], predicted[0], 'S'),
                (o, oracle[1], predicted[1], 'O'), (u, oracle[0] + oracle[1], predicted[0] + predicted[1], 'sum'))):
            ax = axes[a, c]
            if route != 'ordinary_hard' or a == 2:
                label = ('processed observation' if a == 2 else 'matched smoothing target' if route == 'smooth_hard' else 'fixed teacher target')
                ax.plot(time_axis, teacher[0, :, c], label=label, color='black')
            ax.plot(time_axis, dec.reshape(shape)[0, :, c], label='true-ID frozen decode', color='#d98b23', ls='--')
            ax.plot(time_axis, pred.reshape(shape)[0, :, c], label='masked expected decode', color='#177db0')
            ax.fill_between(time_axis, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color='#aaaaaa', alpha=.2)
            ax.set(title=f'{modality} {title}' + (' (arbitrary additive stream)' if route == 'ordinary_hard' and a != 2 else '') + (f' channel {c + 1}' if channels > 1 else ''),
                   xlabel='event time (s)', ylabel='frozen training SD')
            ax.legend(fontsize=7); ax.grid(alpha=.2)
    row = rows[0]
    fig.suptitle(f"Offline processed-observation completion | first heldout identity {row.get('id', row['subject'])}\nGrey: jointly hidden S/O blocks; teacher shown for scoring only", fontsize=9)
    fig.savefig(path, dpi=200); plt.close(fig)


def run_masked_pilot(cfg, run, *, device='cpu', seed=17, route='hard_student'):
    """Train paired/own readers, then frozen-context controls at identical masks.

    CE is interpreted only relative to this route's train-fitted baselines.
    Cross-route comparison uses the common observation-space decoded NMSE.
    """
    if route not in ('hard_student', 'smooth_hard', 'ordinary_hard'):
        raise ValueError('masked pilot requires a frozen deployable hard-ID vocabulary')
    device = str(device)
    if device.startswith('cuda'):
        budget_mib = float(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096))
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(min(budget_mib * 1024 ** 2 / properties.total_memory, 1.), device)
        torch.cuda.reset_peak_memory_stats(device)
    run = Path(run); cache, manifest, all_arrays = _cache(run); all_rows = manifest['rows']
    config = dict(cfg.get('masked', {})); config['vocab_size'] = int(cfg['model']['codebook_size'])
    job = _job(cfg, run, route, seed)
    job_record = json.loads((job / 'summary.json').read_text()) if (job / 'summary.json').exists() else {}
    if job_record.get('status') != 'completed' or job_record.get('step') != int(cfg.get('training', {}).get('steps', 3000)):
        raise ValueError('masked reader requires a completed fixed-budget frozen tokenizer job')
    with np.load(job / 'predictions.npz', allow_pickle=False) as export:
        ids = np.stack([np.stack([export[m + '_ids_' + c] for c in BRANCHES], axis=-1) for m in MODALITIES], axis=-2)
    if ids.dtype.kind not in 'iu' or ids.shape != (len(all_rows), int(manifest.get('patches', 10)), 2, 2):
        raise ValueError('exact same-ROI [N,10,2,2] hard ID export required')
    with np.load(job / 'codewords.npz', allow_pickle=False) as books:
        decoded = {(m, c):books[modality + '_' + branch] for m, modality in enumerate(MODALITIES) for c, branch in enumerate(BRANCHES)}
    if any(len(w) != config['vocab_size'] for w in decoded.values()):
        raise ValueError('vocabulary/decoder count mismatch')
    if np.any(ids >= config['vocab_size']):
        raise ValueError('token IDs exceed frozen vocabulary')
    digest = hashlib.sha256(np.ascontiguousarray(ids).tobytes())
    for key in sorted(decoded): digest.update(np.ascontiguousarray(decoded[key]).tobytes())
    config['frozen_representation_signature'] = digest.hexdigest()
    # Only metadata keys permitted in the baseline; condition/task are ignored.
    baseline_rows = [dict(row, roi=_region(row)) for row in all_rows]
    baseline = fit_token_baselines(ids, baseline_rows, config['vocab_size'])
    output = run / 'masked' / route / ('seed_' + str(seed)); output.mkdir(parents=True, exist_ok=True)
    models = {name:_fit_reader(ids, all_rows, config, output / name, device=device,
                seed=seed, reader=name) for name in ('paired', 'own')}
    selected = np.array([r['split'] in ('validation', 'test') for r in all_rows])
    eval_ids = ids[selected]; rows = [r for i, r in enumerate(all_rows) if selected[i]]
    if not len(rows): raise ValueError('heldout scoring support missing')
    arrays = {key:np.asarray(value[selected]) for key, value in all_arrays.items()}
    if route == 'smooth_hard':
        smoothing = json.loads((cache / 'smoothing.json').read_text())
        for m in MODALITIES:
            for name in ('s', 'e'):
                arrays[m + '_' + name] = np.load(cache / smoothing['arrays'][m + '_' + name], mmap_mode='r')[selected]
    baseline_eval_rows = [dict(r, roi=_region(r)) for r in rows]
    marginal = baseline_probabilities(baseline, baseline_eval_rows)
    metadata = baseline_probabilities(baseline, baseline_eval_rows, metadata=True)
    train = np.array([r['split'] == 'train' for r in all_rows])
    scores, donor_reports = [], []
    for target_m, modality in enumerate(MODALITIES):
        source_m = 1 - target_m
        for task in ('time', 'modality'):
            visible, target = block_masks(eval_ids >= 0, rng=np.random.default_rng(seed + target_m * 10 + (task == 'modality')),
                target_modality=target_m, block_patches=int(config.get('block_patches', 2)), task=task)
            contexts = [('paired', models['paired'], eval_ids, visible),
                        ('own_reader', models['own'], eval_ids, own_context(visible, target_m)),
                        ('frozen_remove_source_modality', models['paired'], eval_ids, own_context(visible, target_m))]
            for c, branch in enumerate(BRANCHES):
                ablated = visible.copy(); ablated[:, :, source_m, c] = False
                contexts.append(('frozen_remove_source_' + branch, models['paired'], eval_ids, ablated))
            # Retain only source tokens preceding each sample's first target patch.
            past = visible.copy()
            target_time = target[:, :, target_m].any(-1)
            for i in range(len(rows)):
                locations = np.flatnonzero(target_time[i])
                if len(locations): past[i, locations[0]:] = False
            contexts.append(('frozen_past_only', models['paired'], eval_ids, past))
            donor, available, report = matched_donor_tokens(eval_ids, rows, decoded, source_modality=source_m,
                training_ids=ids[train], training_rows=[r for i, r in enumerate(all_rows) if train[i]], seed=seed)
            donor_visible = visible.copy(); donor_visible[:, :, source_m] &= available[:, :, None]
            contexts.append(('frozen_matched_donor', models['paired'], donor, donor_visible))
            donor_reports.append(dict(target_modality=modality, task=task, **report))
            # Nonwrapping time shift: vacated source tokens are hidden, never wrapped.
            shift = min(int(config.get('shift_patches', 3)), ids.shape[1] - 1)
            shifted, shifted_visible = nonwrapping_source_shift(eval_ids, visible, source_m, shift)
            contexts.append(('frozen_nonwrapping_source_shift', models['paired'], shifted, shifted_visible))
            # Explicit train-fit probability controls, with no contextual input.
            controls = [('training_marginal', marginal), ('training_metadata', metadata)]
            for name, model, context_ids, context_visible in contexts:
                probability = _predict(model, context_ids, context_visible, device=device,
                    batch_size=int(config.get('eval_batch_size', 128)))
                window_scores, predicted = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                    marginal, metadata, route=route, seed=seed, task=task, context=name, target_modality=target_m)
                scores.extend(_metric_subjects(window_scores))
                if name == 'paired' and task == 'modality':
                    oracle = {c:decoded[(target_m, c)][np.maximum(eval_ids[:, :, target_m, c], 0)] for c in range(2)}
                    _trace_plot(output / ('trace_' + modality + '.png'), rows, arrays, oracle, predicted,
                                target[:, :, target_m].all(-1), modality, route)
            for name, probability in controls:
                window_scores, _ = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                    marginal, metadata, route=route, seed=seed, task=task, context=name, target_modality=target_m)
                scores.extend(_metric_subjects(window_scores))
            # Waveform baselines see the same visible token interface as readers.
            oracle_sum = sum(decoded[(target_m, c)][np.maximum(eval_ids[:, :, target_m, c], 0)] for c in range(2))
            mean_wave = np.asarray(all_arrays[modality + '_u'][train]).mean(axis=0)
            selected_target = target[:, :, target_m].all(-1)
            for name in ('training_mean_waveform', 'linear_interpolation_visible_decodes'):
                prediction = np.broadcast_to(mean_wave, oracle_sum.shape).copy()
                if name.startswith('linear'):
                    for i in range(len(rows)):
                        known = np.flatnonzero(visible[i, :, target_m].all(-1))
                        if len(known):
                            for d in range(oracle_sum.shape[-1]):
                                prediction[i, :, d] = np.interp(np.arange(ids.shape[1]), known, oracle_sum[i, known, d])
                error = np.mean((prediction - arrays[modality + '_u']) ** 2, axis=-1)
                window_rows = [dict(dataset=r.get('dataset', ''), subject=r['subject'],
                    record=r.get('record', r.get('join_key', '')), split=r['split'], variant=route,
                    seed=seed, task=task, context=name, modality=modality, component='sum', metric='NMSE', target_family='processed_observation',
                    value=float(error[i][selected_target[i]].mean()), failures=0)
                    for i, r in enumerate(rows) if selected_target[i].any()]
                scores.extend(_metric_subjects(window_rows))
        # A same-position opposite branch remains visible: explicitly diagnostic.
        for c, branch in enumerate(BRANCHES):
            visible, target = block_masks(eval_ids >= 0, rng=np.random.default_rng(seed + 100 + target_m),
                target_modality=target_m, block_patches=int(config.get('block_patches', 2)), task='modality', single_branch=c)
            probability = _predict(models['paired'], eval_ids, visible, device=device)
            window_rows, _ = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                marginal, metadata, route=route, seed=seed, task='single_branch_' + branch + '_diagnostic',
                context='paired', target_modality=target_m)
            scores.extend(_metric_subjects(window_rows))
    _csv(output / 'subject_metrics.csv', scores)
    _json(output / 'donor_controls.json', donor_reports)
    summary = dict(schema='dual_component_masked_pilot_v1', status='completed', route=route, seed=seed,
        reader_steps=int(config.get('steps', 1000)), readers=['paired', 'own'],
        train_rows=int(train.sum()), evaluation_rows=len(rows), subject_metric_rows=len(scores),
        teacher_inputs=False, sample_input='hard integer IDs only, one ROI 10-patch context',
        decoded_expectation='probability-weighted individually decoded fixed codewords',
        cross_vocabulary_CE_ranking_allowed=False, main_comparison='same-observation-space sum NMSE',
        ordinary_branch_semantics='arbitrary additive streams' if route == 'ordinary_hard' else None,
        context_controls='frozen reader ablations are dependencies, not causal physiological effects',
        physical_compatibility=dict(status='not_evaluated',
            reason='Boundary jumps describe morphological continuity only; no Tak whole-trajectory compatibility witness was evaluated'),
        continuity_diagnostics='S patch-boundary jump roughness and signed-jump error MSE in training-SD squared; morphological description, not Tak reachability or compatibility',
        donor_reports=donor_reports, scope='offline processed-observation completion',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
        cuda_memory_budget_mib=cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096) if device.startswith('cuda') else None,
        inference='descriptive heldout public-development evidence; no protected confirmation')
    _json(output / 'summary.json', summary)
    return summary


def vocabulary_quality_screen(subjects, cfg):
    settings = cfg.get('evaluation', {}).get('quality_screen', {})
    minimum_gain = float(settings.get('minimum_relative_gain_over_train_mean', .1))
    maximum_gap = float(settings.get('maximum_hard_minus_continuous_over_mean', .1))
    checks = []
    for dataset in sorted({r.get('dataset', '') for r in subjects}):
        for modality in MODALITIES:
            for branch in BRANCHES:
                values = {}
                for variant in ('hard_student', 'continuous_student', 'training_mean'):
                    bysubject = defaultdict(list)
                    for row in subjects:
                        if (row.get('dataset', '') == dataset and row['modality'] == modality
                                and row['component'] == branch and row.get('signal') in ('EEG_proxy', 'Hb_joint') and row['split'] == 'validation'
                                and row['variant'] == variant and row.get('NMSE') is not None):
                            bysubject[row['subject']].append(row['NMSE'])
                    values[variant] = float(np.mean([np.mean(x) for x in bysubject.values()])) if bysubject else None
                hard, continuous, mean = (values[k] for k in ('hard_student', 'continuous_student', 'training_mean'))
                available = all(x is not None and np.isfinite(x) for x in (hard, continuous, mean)) and mean > 0
                gain = 1 - hard / mean if available else None
                gap = (hard - continuous) / mean if available else None
                checks.append(dict(dataset=dataset, modality=modality, component=branch, validation_NMSE=values,
                    relative_gain_over_train_mean=gain, hard_minus_continuous_over_mean=gap,
                    passed=bool(available and gain >= minimum_gain and gap <= maximum_gap)))
    passed = bool(checks and all(x['passed'] for x in checks))
    return dict(passed=passed, status='passed' if passed else 'diagnostic_only', checks=checks,
        thresholds=dict(minimum_relative_gain_over_train_mean=minimum_gain,
                        maximum_hard_minus_continuous_over_mean=maximum_gap),
        on_failure='continue fixed small masked pilot diagnostic-only; no scale expansion',
        selection_split='validation only; no test-based training or selection')


def analyze_teacher_sensitivity(cfg, run):
    """Paired equal-budget training-identity sensitivity, never a truth audit."""
    root = Path(run) / 'sensitivity'
    declared = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else {}
    manifest_path = root / 'manifest.json'
    seed = int(cfg.get('sensitivity', {}).get('seed', 17))
    dirs = {name:root / name / ('seed_' + str(seed)) for name in ('reference', 'alternative')}
    expected_steps = int(cfg.get('sensitivity', {}).get('paired_steps', 500))
    arm_records = [json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {} for path in dirs.values()]
    ready = all(record.get('status') == 'completed' and record.get('step') == expected_steps for record in arm_records)
    if not manifest_path.exists() or not ready or any(not (path / 'codewords.npz').exists() for path in dirs.values()):
        return dict(status='unavailable', reason=declared.get('reason', 'No complete legal near-equivalent whole-trajectory witness and paired equal-budget retraining exports'),
                    finite_search=declared, physiological_truth_claim=False)
    manifest = json.loads(manifest_path.read_text()); rows = manifest['rows']
    exports, codewords = {}, {}
    for name, path in dirs.items():
        with np.load(path / 'predictions.npz', allow_pickle=False) as data:
            exports[name] = {key:data[key] for key in data.files}
        with np.load(path / 'codewords.npz', allow_pickle=False) as data:
            codewords[name] = {key:data[key] for key in data.files}
    results = []
    for modality in MODALITIES:
        for branch in BRANCHES:
            key = modality + '_' + branch
            mapping, effect = align_codewords(codewords['reference'][key], codewords['alternative'][key])
            ref = exports['reference'][modality + '_ids_' + branch]
            alt = exports['alternative'][modality + '_ids_' + branch]
            valid = (ref >= 0) & (alt >= 0)
            delta = np.mean((exports['reference'][key] - exports['alternative'][key]) ** 2, axis=-1)
            for dataset in sorted({r.get('dataset', '') for r in rows}):
                select = np.array([r.get('dataset', '') == dataset for r in rows])
                keep = select[:, None] & valid
                results.append(dict(dataset=dataset, modality=modality, component=branch, **effect,
                    assigned_decoded_waveform_change_NMSE=float(delta[keep].mean()) if keep.any() else None,
                    aligned_ID_agreement=float(np.mean(ref[keep] == mapping[alt[keep]])) if keep.any() else None,
                    windows=int(select.sum()), subjects=len({r['subject'] for i, r in enumerate(rows) if select[i]}),
                    interpretation='paired train-identity teacher sensitivity; no heldout-generalization or physiological-truth claim'))
    _csv(root / 'prototype_sensitivity.csv', results)
    result = dict(status='completed', seed=seed, paired_steps=int(cfg.get('sensitivity', {}).get('paired_steps', 500)),
        rows=results, evidence=declared, evaluation='same selected training identities, equal optimizer budget',
        scientific_limit='finite legal whole-trajectory witness, not posterior or complete feasible set',
        physiological_truth_claim=False)
    _json(root / 'analysis.json', result)
    return result


def summarize_masked_pilots(cfg, run):
    run = Path(run); config = cfg.get('masked', {}); subjects, missing = [], []
    for route in config.get('routes', ('hard_student', 'smooth_hard', 'ordinary_hard')):
        for seed in config.get('seeds', cfg.get('seeds', (17, 29, 43))):
            path = run / 'masked' / route / ('seed_' + str(seed)) / 'subject_metrics.csv'
            if not path.exists():
                missing.append(dict(route=route, seed=seed)); continue
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    row['value'] = float(row['value']) if row['value'] else None
                    row['seed'] = int(row['seed']); subjects.append(row)
    if not subjects: return dict(status='not_run', missing=missing)
    comparisons = []; samples = int(cfg.get('evaluation', {}).get('bootstrap_repeats', 1000))
    grouped = defaultdict(list)
    for row in subjects:
        grouped[(row['variant'], row['dataset'], row['split'], row['task'], row['modality'], row['component'], row['metric'])].append(row)
    for (route, dataset, split, task, modality, component, metric), group in grouped.items():
        if metric not in ('NMSE', 'NMSE_HbO', 'NMSE_HbR', 'CE_nats', 'DeltaCE_metadata_nats'): continue
        first = [r for r in group if r['context'] == 'paired']
        for context in sorted({r['context'] for r in group} - {'paired'}):
            second = [r for r in group if r['context'] == context]
            result = paired_subject_bootstrap(first, second, value='value', samples=samples, seed=int(cfg.get('seed', 0)))
            result['effect'] = 'paired minus control; ' + ('higher favors paired' if metric.startswith('DeltaCE') else 'lower favors paired')
            comparisons.append(dict(route=route, dataset=dataset, split=split, task=task, modality=modality,
                component=component, metric=metric, first='paired', second=context, **result))
    # Across vocabularies compare only the shared observation-space sum error.
    for control in ('smooth_hard', 'ordinary_hard'):
        keys = {(r['dataset'], r['split'], r['task'], r['modality']) for r in subjects
                if r['metric'] in ('NMSE', 'NMSE_HbO', 'NMSE_HbR') and r['component'] == 'sum' and r['context'] == 'paired'}
        for dataset, split, task, modality in keys:
            def pick(route):
                return [r for r in subjects if r['variant'] == route and r['dataset'] == dataset
                    and r['split'] == split and r['task'] == task and r['modality'] == modality
                    and r['metric'] == 'NMSE' and r['component'] == 'sum' and r['context'] == 'paired']
            result = paired_subject_bootstrap(pick('hard_student'), pick(control), value='value', samples=samples, seed=int(cfg.get('seed', 0)))
            comparisons.append(dict(route='cross_vocabulary', dataset=dataset, split=split, task=task,
                modality=modality, component='sum', metric='NMSE', first='hard_student', second=control, **result))
    _csv(run / 'analysis' / 'masked_subject_metrics.csv', subjects)
    _csv(run / 'analysis' / 'masked_paired_comparisons.csv', comparisons)
    result = dict(status='completed' if not missing else 'partial', missing=missing, subject_rows=len(subjects),
        comparisons=comparisons, cross_vocabulary_CE_ranking_allowed=False,
        aggregation='equal record means per subject; seeds averaged within subject before resampling',
        inference='exploratory percentile subject bootstrap; no multiple-comparison significance claims')
    _json(run / 'analysis' / 'masked_summary.json', result)
    return result
