"""Bounded train-subject comparison of fixed and integer-driven adaptive gain."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics, joint_reconstruction_summary
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, attributes, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_joint_eeg, base_tokenizer_batch
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier
from src.tokenizers.backward_adaptive_carrier import BackwardAdaptiveCarrier, decode_sequences, quantize_sequences, load_model
from src.tokenizers.spatial_waveform_dictionary import spectral_features
from audit_joint_tokenizer_patterns import write_json

FIELDS = {'power_4_8': (0, 1), 'power_8_13': (1, 2), 'rms_envelope': (2, 10)}
PRIOR_CPU = float(os.environ.get('TOKENIZER_BACKWARD_PRIOR_CPU_SECONDS', '0'))


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def budget(cfg):
    if PRIOR_CPU + time.process_time() >= cfg['resources']['main_CPU_budget_seconds']:
        raise RuntimeError('backward-adaptive fixed experiment CPU budget reached')


def prepare(cfg, rows, scales, source, out):
    directory = ROOT/cfg['source']['cached_native_run']
    panel = json.loads((directory/'panel.json').read_text())
    # The previously fitted neural holdout labels are irrelevant to this cache
    # consumer. Its data identities and all 400 original train crops are retained.
    folds = json.loads((ROOT/cfg['source']['basis_run']/'fold_subjects.json').read_text())
    sites = sorted({(r['dataset'], p) for r in rows if r['split'] == 'train' and r['modality'] == 'eeg' for p in r['positions']})
    site_index = {key: j for j, key in enumerate(sites)}
    for meta in panel['cases']:
        row = rows[meta['crop']['row']]
        if (row['split'] != 'train' or row['modality'] != 'eeg'
                or row['dataset'] != meta['dataset'] or row['subject'] != meta['subject']):
            raise ValueError('cached native panel violates train boundary before arrays')
    cases = []
    score_cache = OrderedDict()
    for n, meta in enumerate(panel['cases']):
        budget(cfg)
        item = torch.load(directory/'cache'/f'{n:04d}.pt', map_location='cpu', weights_only=False)
        crop, dataset = meta['crop'], meta['dataset']
        expected = np.asarray([(crop['row'], crop['start_s'], p) for p in crop['positions']])
        np.testing.assert_array_equal(item['examples'], expected)
        names = source.metadata['positions'][dataset]
        if item['x'].shape != (len(names), 15, 400):
            raise ValueError('cached montage layout mismatch')
        native = item['native']
        batch = base_tokenizer_batch(rows, item['examples'], scales, array_cache=score_cache, cache_limit=16)
        for field in ('s', 'o', 'mask'):
            np.testing.assert_array_equal(batch[field], item[field][native])
        np.testing.assert_array_equal(batch['x'], item['score_x'])
        complete = batch['mask'].all(-1)
        target = {}
        for b in ('S', 'O'):
            raw, slices = attributes(batch[b.lower()][complete], 'eeg')
            target[b] = np.zeros((*complete.shape, 10))
            target[b][complete] = np.concatenate([raw[:, slice(*slices[k])] for k in FIELDS], axis=1)
        row = rows[crop['row']]
        loc = np.array([site_index[dataset, row['positions'][p]] for p in crop['positions']])
        background = 4*loc[:, None] + np.searchsorted([.5, 1., 2.], np.sqrt(np.mean(batch['x']**2, axis=-1)), side='right')
        full_complete = np.zeros((len(names), 15), bool)
        full_complete[native] = complete
        cases.append(dict(dataset=dataset, subject=meta['subject'], fold=folds[dataset][meta['subject']], crop=crop,
            x=item['x'], observed=item['observed'][None], native=native, examples=item['examples'], scales=item['scales'],
            batch=batch, complete=complete, full_complete=full_complete, target=target, background=background))
    if len(cases) != 400 or sum(len(c['examples']) for c in cases) != 17040:
        raise ValueError('fixed 400 crop and 17040 position population required')
    write_json(out/'population.json', dict(crops=400, scoring_position_windows=17040,
        subjects=sorted({(c['dataset'], c['subject']) for c in cases}), complete_patches=sum(int(c['complete'].sum()) for c in cases),
        folds=folds, cases=[{k: c[k] for k in ('dataset', 'subject', 'fold', 'crop', 'examples')} for c in cases],
        original_model_exposure=cfg['training']['exposure'], cached_neural_holdout_partition_used=False,
        validation_reads=0, teacher_refits=0))
    return cases, len(sites)*4


def make_groups(source, cases, fold, cfg, out):
    for n, c in enumerate(cases):
        budget(cfg)
        physical = source.continuous_source(c['x'][None])
        c['carriers'] = source.analyze(physical, c['dataset'], c['observed'])
        if n % 40 == 0:
            write_json(out/'progress.json', dict(stage='frozen_carriers', fold=fold, completed=n+1, planned=len(cases), CPU_seconds=time.process_time()))
    groups = {}
    initial = {(d, b): np.zeros(len(names)) for d, names in source.metadata['positions'].items() for b in ('S', 'O')}
    for branch in ('S', 'O'):
        for band in range(5):
            values, masks, info = [], [], []
            for n, c in enumerate(cases):
                positions = np.flatnonzero(source.roles(c['dataset']) == band)
                values.append(c['carriers'][branch][0, positions])
                masks.append(c['full_complete'][positions])
                info.extend((n, int(p)) for p in positions)
            sequence = np.concatenate(values)
            complete = np.concatenate(masks)
            information = np.asarray(info)
            fitting = np.asarray([cases[n]['fold'] != fold for n, _ in info])
            rms = np.sqrt(np.mean(sequence**2, axis=-1))
            fallback = max(float(np.median(rms[complete & fitting[:, None]])), 1e-8)
            scale = np.empty(len(info))
            for dataset, names in source.metadata['positions'].items():
                for position in np.flatnonzero(source.roles(dataset) == band):
                    chosen = np.asarray([cases[n]['dataset'] == dataset and p == position and cases[n]['fold'] != fold for n, p in info])
                    usable = complete & chosen[:, None]
                    value = max(float(np.median(rms[usable])), 1e-8) if usable.any() else fallback
                    initial[dataset, branch][position] = value
            for j, (n, position) in enumerate(info):
                scale[j] = initial[cases[n]['dataset'], branch][position]
            features = spectral_features(sequence.reshape(-1, 400), band).reshape(len(sequence), 15, -1)
            rng = np.random.default_rng(cfg['training']['seed'] + band)
            fit_patch = np.zeros(complete.shape, bool)
            people = sorted({(cases[n]['dataset'], cases[n]['subject']) for n, _ in info if cases[n]['fold'] != fold})
            for person in people:
                unit = np.asarray([(cases[n]['dataset'], cases[n]['subject']) == person for n, _ in info])
                rows, patches = np.nonzero(complete & fitting[:, None] & unit[:, None])
                take = rng.choice(len(rows), min(len(rows), cfg['training']['max_fit_patches_per_subject_branch_band']), replace=False)
                fit_patch[rows[take], patches[take]] = True
            ratios = rms[fit_patch]/np.broadcast_to(scale[:, None], rms.shape)[fit_patch]
            adjacent = complete[:, 1:] & complete[:, :-1] & fitting[:, None]
            steps = np.abs(np.log(np.maximum(rms[:, 1:], 1e-12)/np.maximum(rms[:, :-1], 1e-12)))[adjacent]
            settings = dict(minimum_ratio=float(np.clip(np.quantile(ratios, .001), 1/32, 1)),
                maximum_ratio=float(np.clip(np.quantile(ratios, .999), 1, 32)),
                maximum_log_step=float(np.clip(np.quantile(steps, .90), np.log(1.1), np.log(2))))
            groups[branch, band] = dict(sequence=sequence, features=features, complete=complete, info=information,
                fitting=fitting, fit_patch=fit_patch, initial=scale, settings=settings,
                observed=np.asarray([bool(cases[n]['observed'][0, p]) for n, p in info]))
    return groups, initial


def update_atoms(group, codes, gains, positive):
    valid = group['fit_patch']
    index = codes[valid]-1
    if np.any(index < 0):
        raise ValueError('missing observations entered atomic training')
    gain = gains[valid]
    numerator = np.zeros((positive, 400))
    denominator = np.zeros(positive)
    np.add.at(numerator, index, group['sequence'][valid]*gain[:, None])
    np.add.at(denominator, index, gain**2)
    count = np.bincount(index, minlength=positive)
    return numerator, denominator, count


def fit_group(key, group, cfg, out, fold):
    from sklearn.cluster import MiniBatchKMeans
    branch, band = key
    path = out/f'fold{fold}_{branch}_band{band}.npz'
    metadata_path = path.with_suffix('.json')
    if path.exists() and metadata_path.exists():
        with np.load(path, allow_pickle=False) as saved:
            books = {arm: {k: saved[f'{arm}_{k}'].copy() for k in ('atoms', 'centers', 'counts')} for arm in cfg['model']['arms']}
        return key, books, json.loads(metadata_path.read_text())
    budget(cfg)
    positive = cfg['model']['positive_atoms_per_branch_band']
    valid = group['fit_patch']
    scaled = (group['features']/group['initial'][:, None, None])[valid].astype('f4')
    km = MiniBatchKMeans(n_clusters=positive, batch_size=cfg['training']['kmeans_batch_size'], max_iter=cfg['training']['kmeans_iterations'],
        random_state=cfg['training']['seed']+band, n_init=1, reassignment_ratio=0., max_no_improvement=6)
    labels = km.fit_predict(scaled)
    all_codes = np.ones(group['complete'].shape, np.int32)
    all_codes[valid] = labels+1
    initial_gain = np.broadcast_to(group['initial'][:, None], all_codes.shape).copy()
    numerator, denominator, count = update_atoms(group, all_codes, initial_gain, positive)
    atoms = np.zeros((positive+1, 400))
    used = denominator > 0
    atoms[1:][used] = numerator[used]/denominator[used, None]
    for code in np.flatnonzero(~used):
        distance = np.square(scaled-km.cluster_centers_[code]).sum(1)
        candidates = (group['sequence']/group['initial'][:, None, None])[valid]
        atoms[code+1] = candidates[int(distance.argmin())]
    base = dict(atoms=atoms, centers=spectral_features(atoms[1:], band), counts=np.r_[0, count])
    atom_rms = np.sqrt(np.mean(atoms[1:]**2, axis=1))
    order = np.argsort(atom_rms)
    weights = count[order]
    reference = atom_rms[order[np.searchsorted(np.cumsum(weights), weights.sum()/2)]]
    settings = dict(group['settings'], reference_atom_RMS=max(float(reference), 1e-8))
    books, traces = {}, {}
    fit_sequence = group['fitting']
    for arm in cfg['model']['arms']:
        book = copy.deepcopy(base)
        history = []
        for iteration in range(cfg['training']['quantizer_refit_iterations']):
            budget(cfg)
            codes, gains = quantize_sequences(group['features'], group['initial'], book, settings, arm == 'backward_adaptive', group['observed'])
            numerator, denominator, count = update_atoms(group, codes, gains, positive)
            updated = book['atoms'].copy()
            occupied = denominator > 0
            updated[1:][occupied] = numerator[occupied]/denominator[occupied, None]
            book = dict(atoms=updated, centers=spectral_features(updated[1:], band), counts=np.r_[0, count])
            decoded, replay_gain = decode_sequences(codes, group['initial'], book, settings, arm == 'backward_adaptive')
            # Atom updates alter the integer state response. This diagnostic
            # records that changed training objective; final IDs are recomputed.
            error = np.square((decoded-group['sequence'])[valid]).sum()
            energy = np.square(group['sequence'][valid]).sum()
            history.append(dict(iteration=iteration+1, carrier_MSE_over_energy=float(error/max(energy, 1e-12)),
                active_codes=int(occupied.sum()), fit_complete_patches=int(valid.sum()),
                final_patch_gain_ratio_median=float(np.median(replay_gain[fit_sequence, -1]/group['initial'][fit_sequence]))))
        books[arm], traces[arm] = book, history
    payload = {f'{arm}_{k}': value for arm, book in books.items() for k, value in book.items()}
    temporary = path.with_suffix('.pending.npz')
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    record = dict(settings=settings, traces=traces, initial_books_identical=True, fit_patch_indices_identical=True,
                  shared_initial_atom_sha256=hashlib.sha256(base['atoms'].tobytes()).hexdigest(),
                  fit_patch_mask_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
                  schema='backward_adaptive_book_v1', fold=fold, branch=branch, band=band)
    write_json(metadata_path, record)
    return key, books, record


def assign_all(model, cases):
    return [model.assign(c['carriers'], c['dataset'], c['observed']) for c in cases]


def readout(model, cases, codes, fit, held, backgrounds):
    audits = {b: SparsePatternAudit(5*model.codebook_size, backgrounds, [], 10, FIELDS) for b in ('S', 'O')}
    people = {b: defaultdict(set) for b in ('S', 'O')}
    moments = {b: np.zeros((2, 10)) for b in ('S', 'O')}
    for i in fit:
        c = cases[i]
        band = model.roles(c['dataset'])[c['native']][:, None]
        for b in ('S', 'O'):
            ids = (codes[i][b][0, c['native']]+band*model.codebook_size)[c['complete']]
            y = c['target'][b][c['complete']]
            audits[b].fit(ids, c['background'][c['complete']], y)
            moments[b][0] += y.sum(0)
            moments[b][1] += (y*y).sum(0)
            for code in np.unique(ids):
                people[b][int(code)].add((c['dataset'], c['subject']))
    errors, support = {b: [] for b in ('S', 'O')}, {b: [0, 0] for b in ('S', 'O')}
    held_count = {b: np.zeros(5*model.codebook_size, int) for b in ('S', 'O')}
    for i in held:
        c = cases[i]
        band = model.roles(c['dataset'])[c['native']][:, None]
        valid = c['complete']
        for b in ('S', 'O'):
            ids = (codes[i][b][0, c['native']]+band*model.codebook_size)[valid]
            background, _, conditional = audits[b].read(ids, c['background'][valid])
            y = c['target'][b][valid]
            error = np.zeros((*valid.shape, 2, 3))
            error[valid] = np.stack([feature_means((p-y)**2, FIELDS) for p in (background, conditional)], axis=1)
            means = error.sum(1)/valid.sum(1).clip(1)[:, None, None]
            means[~valid.any(1)] = np.nan
            errors[b].append(means)
            strong = (audits[b].local.count[ids] >= 8) & np.asarray([len(people[b][int(k)]) >= 2 for k in ids])
            held_count[b] += np.bincount(ids, minlength=5*model.codebook_size)
            support[b][0] += int(strong.sum())
            support[b][1] += len(ids)
    variance = {}
    for b in ('S', 'O'):
        count = audits[b].local.count.sum()
        variance[b] = feature_means((moments[b][1]/count-(moments[b][0]/count)**2)[None], FIELDS)[0]
    usage = [dict(branch=b, band=band, ID=code, train_complete_patches=int(audits[b].local.count[band*model.codebook_size+code]),
                  train_subjects=len(people[b][band*model.codebook_size+code]), held_complete_patches=int(held_count[b][band*model.codebook_size+code]))
             for b in ('S', 'O') for band in range(5) for code in range(model.codebook_size)]
    return audits, variance, {b: np.concatenate(v) for b, v in errors.items()}, support, usage


def score_waveforms(model, cases, codes, held, rows, out, arm, fold):
    records, support, examples = defaultdict(list), defaultdict(list), []
    gain_trace = []
    for ordinal, i in enumerate(held):
        c = cases[i]
        hard = model.decode_ids(codes[i], c['dataset'])
        carrier, gain = model.decode_carriers(codes[i], c['dataset'], return_gains=True)
        for b in ('S', 'O'):
            initial = model.initial_scales[c['dataset'], b][:, None]
            ratio = gain[b][0]/initial
            target_rms = np.sqrt(np.mean(c['carriers'][b][0]**2, axis=-1))
            predicted_rms = np.sqrt(np.mean(carrier[b][0]**2, axis=-1))
            gain_trace.append(dict(case=i, branch=b, ratio=ratio[c['native']],
                carrier_target_RMS=target_rms[c['native']], decoded_carrier_RMS=predicted_rms[c['native']],
                complete=c['complete']))
        audit = joint_reconstruction_metrics({k: v[0, c['native']] for k, v in hard.items()}, c['batch'], 1, c['scales'])
        for key, value in audit['metrics'].items():
            records[key].append(value)
        for key, value in audit['support'].items():
            support[key].append(value)
        examples.extend(c['examples'])
        if ordinal % 25 == 0:
            write_json(out/'progress.json', dict(stage='hard_full_training_score', arm=arm, fold=fold, completed=ordinal+1, planned=len(held), CPU_seconds=time.process_time()))
    return np.asarray(examples), {k: np.concatenate(v) for k, v in records.items()}, {k: np.concatenate(v) for k, v in support.items()}, gain_trace


def natural_context(model, audits, variance, cases, codes, held, rows, scales):
    chosen = {}
    for i in held:
        c = cases[i]
        row = rows[c['crop']['row']]
        adjacent = c['crop']['start_s']+2
        if not any(v['start_s'] == adjacent for v in row['inputs']['train']):
            continue
        key = c['dataset'], c['subject']
        if key not in chosen or (c['crop']['row'], c['crop']['start_s']) < (cases[chosen[key]]['crop']['row'], cases[chosen[key]]['crop']['start_s']):
            chosen[key] = i
    record = []
    cache = OrderedDict()
    for i in chosen.values():
        c = cases[i]
        row = rows[c['crop']['row']]
        names = model.metadata['positions'][c['dataset']]
        shifted = dict(c['crop'], start_s=c['crop']['start_s']+2, positions=[])
        fields, observed, _ = base_tokenizer_joint_eeg(rows, shifted, scales, names, cache, allowed_splits=('train',))
        x = fields['x'].T.astype('f4').reshape(1, len(names), 15, 400)
        common = c['observed'] & observed[None]
        original = model.encode(c['x'][None], c['dataset'], common)
        other = model.encode(x, c['dataset'], common)
        np.testing.assert_array_equal(c['x'][common[0], 7], x[0, common[0], 6])
        band = model.roles(c['dataset'])[c['native']]
        selected = c['complete'][:, 7] & common[0, c['native']]
        entry = dict(dataset=c['dataset'], subject=c['subject'], crop=c['crop'], positions=int(selected.sum()), input_patch_equal=True)
        if not selected.any():
            entry['status'] = 'no_common_complete_center_positions'
            record.append(entry)
            continue
        for b in ('S', 'O'):
            left = original[b][0, c['native'], 7][selected]+band[selected]*model.codebook_size
            right = other[b][0, c['native'], 6][selected]+band[selected]*model.codebook_size
            bg = c['background'][:, 7][selected]
            a = audits[b].read(left, bg)[2]
            z = audits[b].read(right, bg)[2]
            difference = feature_means((z-a)**2, FIELDS).mean(0)
            entry[b] = dict(ID_change_fraction=float(np.mean(left != right)),
                predicted_attribute_change_over_train_variance=difference/np.maximum(variance[b], 1e-20))
        record.append(entry)
    return record


def run(args, cfg):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'summary.json').exists():
        raise FileExistsError('retained completed backward-adaptive run')
    remaining = int(cfg['resources']['CPU_budget_seconds']-PRIOR_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (remaining-20, remaining))
    if cfg['data']['fitting_splits'] != ['train'] or cfg['data']['evaluation_splits'] != ['train'] or any(cfg['data'][k] != 'forbidden' for k in ('validation_access', 'confirmation_access', 'test_access', 'protected_data')):
        raise ValueError('training-only paired experiment required')
    write_json(out/'resolved_config.json', cfg)
    base = yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    panel = json.loads((ROOT/cfg['source']['cached_native_run']/'panel.json').read_text())
    if panel['manifest_identity'] != rank(manifest)[:24] or panel['scales_identity'] != manifest['scales_identity']:
        raise ValueError('original 400 native cache and G0 identity mismatch before arrays')
    basis = ROOT/cfg['source']['basis_run']
    source = AnchoredSpatialCarrier.load(basis/'fold_model_0.npz')
    cases, backgrounds = prepare(cfg, manifest['rows'], scales, source, out)
    aggregate = {arm: dict(metrics=[], support=[], pattern={b: [] for b in ('S', 'O')}, qualified={b: [0, 0] for b in ('S', 'O')}, contexts=[]) for arm in cfg['model']['arms']}
    all_examples = []
    for fold in range(4):
        budget(cfg)
        source_path = basis/f'fold_model_{fold}.npz'
        source = AnchoredSpatialCarrier.load(source_path)
        if source.metadata['input_identity'] != manifest['cfg_identity'] or source.metadata['scales_identity'] != manifest['scales_identity']:
            raise ValueError('frozen source coordinate changed')
        groups, initial = make_groups(source, cases, fold, cfg, out)
        books = {arm: {} for arm in cfg['model']['arms']}
        settings = {}
        with ThreadPoolExecutor(max_workers=cfg['resources']['book_workers']) as pool:
            pending = [pool.submit(fit_group, key, group, cfg, out, fold) for key, group in groups.items()]
            for completed in as_completed(pending):
                key, fitted, report = completed.result()
                settings[key] = report['settings']
                for arm in books:
                    books[arm][key] = fitted[arm]
                write_json(out/'progress.json', dict(stage='fit_matched_books', fold=fold, completed=len(settings), planned=10, CPU_seconds=time.process_time()))
        del groups
        fit = [i for i, c in enumerate(cases) if c['fold'] != fold]
        held = [i for i, c in enumerate(cases) if c['fold'] == fold]
        for arm in cfg['model']['arms']:
            metadata = dict(schema='backward_adaptive_carrier_v1', adaptive=arm == 'backward_adaptive', codebook_size=513,
                seed=cfg['training']['seed'], held_fold=fold, input_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
                fitting_split='train', per_window_floating_decoder_side_information=0, state_reset_each_context=True)
            model = BackwardAdaptiveCarrier(source, books[arm], initial, settings, metadata)
            model_path = out/f'{arm}_fold{fold}.npz'
            if model_path.exists():
                retained_model = load_model(model_path)
                for key in books[arm]:
                    for name in ('atoms', 'centers', 'counts'):
                        np.testing.assert_array_equal(retained_model.books[key][name], model.books[key][name])
                for key in initial:
                    np.testing.assert_array_equal(retained_model.initial_scales[key], initial[key])
                if retained_model.settings != model.settings:
                    raise ValueError('resumed fixed update rule changed')
                model = retained_model
            else:
                model.save(model_path, source_path)
            all_ID_path = out/f'{arm}_fold{fold}_all_IDs.npz'
            if all_ID_path.exists():
                with np.load(all_ID_path, allow_pickle=False) as stored:
                    codes = [{b: stored[f'{b}_{i}'].copy() for b in ('S', 'O')} for i in range(len(cases))]
            else:
                codes = assign_all(model, cases)
            for check in [next(i for i in held if cases[i]['dataset'] == d) for d in sorted({cases[i]['dataset'] for i in held})]:
                c = cases[check]
                encoded, forward_gain = model.assign(c['carriers'], c['dataset'], c['observed'], return_gains=True)
                _, replay_gain = model.decode_carriers(encoded, c['dataset'], return_gains=True)
                for b in ('S', 'O'):
                    np.testing.assert_array_equal(encoded[b], codes[check][b])
                    np.testing.assert_array_equal(forward_gain[b], replay_gain[b])
            state_parts = {(b, k): [] for b in ('S', 'O') for k in range(5)}
            for i in fit:
                c = cases[i]
                _, gains = model.decode_carriers(codes[i], c['dataset'], return_gains=True)
                for b in ('S', 'O'):
                    ratio = gains[b][0]/model.initial_scales[c['dataset'], b][:, None]
                    for k in range(5):
                        where = (model.roles(c['dataset']) == k) & c['observed'][0]
                        state_parts[b, k].append(ratio[where])
            write_json(out/f'{arm}_fold{fold}_fit_state_distribution.json',
                {f'{b}_{k}': dict(first_patch_quantiles=np.quantile(np.concatenate(values)[:, 0], [.1, .5, .9, .99]),
                                 last_patch_quantiles=np.quantile(np.concatenate(values)[:, -1], [.1, .5, .9, .99]),
                                 all_patch_quantiles=np.quantile(np.concatenate(values), [.001, .1, .5, .9, .999]),
                                 settings=settings[b, k]) for (b, k), values in state_parts.items()})
            held_ids = {f'{b}_{i}': codes[i][b] for i in held for b in ('S', 'O')}
            np.savez_compressed(out/f'{arm}_fold{fold}_held_IDs.npz', **held_ids)
            # Reload both the model and stored integers before scoring.
            reloaded = load_model(model_path)
            with np.load(out/f'{arm}_fold{fold}_held_IDs.npz', allow_pickle=False) as stored:
                for i in held:
                    loaded = {b: stored[f'{b}_{i}'] for b in ('S', 'O')}
                    expected = model.decode_ids(codes[i], cases[i]['dataset'])
                    repeated = reloaded.decode_ids(loaded, cases[i]['dataset'])
                    for b in ('S', 'O', 'sum'):
                        np.testing.assert_array_equal(expected[b], repeated[b])
                    codes[i] = {b: loaded[b].copy() for b in ('S', 'O')}
            np.savez_compressed(out/f'{arm}_fold{fold}_all_IDs.npz', **{f'{b}_{i}': codes[i][b] for i in range(len(cases)) for b in ('S', 'O')})
            audits, variance, error, support, usage = readout(model, cases, codes, fit, held, backgrounds)
            with (out/f'{arm}_fold{fold}_code_usage.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(usage[0]))
                writer.writeheader()
                writer.writerows(usage)
            examples, metrics, score_support, trace = score_waveforms(reloaded, cases, codes, held, manifest['rows'], out, arm, fold)
            contexts = natural_context(model, audits, variance, cases, codes, held, manifest['rows'], scales)
            np.savez_compressed(out/f'{arm}_fold{fold}_metrics.npz', examples=examples, **metrics,
                **{'support_'+k: v for k, v in score_support.items()}, **{b+'_pattern_errors': error[b] for b in ('S', 'O')})
            write_json(out/f'{arm}_fold{fold}_context.json', contexts)
            np.savez_compressed(out/f'{arm}_fold{fold}_state_diagnostics.npz',
                **{f'{entry["branch"]}_{entry["case"]}_{key}': entry[key] for entry in trace for key in ('ratio', 'carrier_target_RMS', 'decoded_carrier_RMS', 'complete')})
            current = aggregate[arm]
            current['metrics'].append(metrics)
            current['support'].append(score_support)
            current['contexts'].extend(contexts)
            for b in ('S', 'O'):
                current['pattern'][b].append(error[b])
                current['qualified'][b] = [a+v for a, v in zip(current['qualified'][b], support[b])]
            if arm == cfg['model']['arms'][0]:
                all_examples.extend(examples)
        for c in cases:
            del c['carriers']
    summaries = {}
    for arm, current in aggregate.items():
        metrics = {k: np.concatenate([v[k] for v in current['metrics']]) for k in current['metrics'][0]}
        support = {k: np.concatenate([v[k] for v in current['support']]) for k in current['support'][0]}
        summary = joint_reconstruction_summary(metrics, support, all_examples, manifest['rows'])
        pattern = {}
        for b in ('S', 'O'):
            errors = np.concatenate(current['pattern'][b])
            stats = base_training_aggregate({'nmse': errors}, all_examples, manifest['rows'])['metrics']['nmse']
            mse = np.asarray(stats['macro'])
            gain = 1-mse[1]/mse[0]
            fraction = np.mean([np.asarray(v['value'])[1] < np.asarray(v['value'])[0] for v in stats['by_subject']], axis=0)
            mass = current['qualified'][b][0]/current['qualified'][b][1]
            pattern[b] = dict(gain=dict(zip(FIELDS, gain)), positive_subject_fraction=dict(zip(FIELDS, fraction)),
                support_mass=mass, hierarchy=stats, passed=bool(min(gain[:2]) > 0 and max(gain) >= .05 and min(fraction[:2]) >= .5 and mass >= .9))
        context_summary = {}
        for b in ('S', 'O'):
            grouped = defaultdict(list)
            for entry in current['contexts']:
                if b in entry:
                    grouped[entry['dataset']].append(entry[b]['predicted_attribute_change_over_train_variance'])
            values = np.mean([np.mean(group, axis=0) for group in grouped.values()], axis=0) if grouped else np.full(3, np.nan)
            context_summary[b] = dict(mean_attribute_change_over_fit_variance=values, passed=bool(np.isfinite(values).all() and np.all(values <= .25)),
                people=sum(len(v) for v in grouped.values()), by_dataset={d: np.mean(v, axis=0) for d, v in grouped.items()})
        summaries[arm] = dict(waveform=summary, pattern=pattern, natural_context=current['contexts'], natural_context_summary=context_summary)
        np.savez_compressed(out/f'{arm}_metrics.npz', examples=np.asarray(all_examples), **metrics,
            **{'support_'+k: v for k, v in support.items()}, **{b+'_pattern_errors': np.concatenate(current['pattern'][b]) for b in ('S', 'O')})
    write_json(out/'summary.json', dict(status='completed_fixed_training_comparison', arms=summaries, CPU_seconds=PRIOR_CPU+time.process_time(), current_process_CPU_seconds=time.process_time(), prior_attempt_CPU_seconds=PRIOR_CPU,
        original_score_inputs=len(all_examples), validation_reads=0, confirmation_reads=0, test_reads=0,
        integer_replay='all_held_saved_integer_arrays_after_model_reload_exact', output_units='same_G0_train_input_coordinate_restored_by_retained_position_scale',
        state_rule='only preceding decoded atom RMS updates next patch scale; reset at every 30-second call'))


def review(args, cfg):
    """Fresh process: only stored IDs enter the decoder; X is scoring data."""
    out = args.output.resolve()
    lock = (out/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remaining = int(cfg['resources']['CPU_budget_seconds']-PRIOR_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (remaining-10, remaining))
    if (out/'replay_review.json').exists():
        raise FileExistsError('retained completed independent replay review')
    summary = json.loads((out/'summary.json').read_text())
    base = yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    panel = json.loads((ROOT/cfg['source']['cached_native_run']/'panel.json').read_text())
    if panel['manifest_identity'] != rank(manifest)[:24] or panel['scales_identity'] != manifest['scales_identity']:
        raise ValueError('review cache and G0 identity mismatch before arrays')
    source = AnchoredSpatialCarrier.load(ROOT/cfg['source']['basis_run']/'fold_model_0.npz')
    cases, backgrounds = prepare(cfg, manifest['rows'], scales, source, out)
    examples = np.concatenate([np.concatenate([c['examples'] for c in cases if c['fold'] == f]) for f in range(4)])
    metric_arrays = {}
    for arm in cfg['model']['arms']:
        with np.load(out/f'{arm}_metrics.npz', allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored['examples'], examples)
            metric_arrays[arm] = {k: stored[k].copy() for k in stored.files}
    # Fixed first physical identity, and the maximum actual adaptive three-component
    # mean NMSE within each dataset. Neither selection drops failed tail examples.
    identity = {tuple(v): j for j, v in enumerate(examples)}
    chosen = []
    for dataset in sorted({c['dataset'] for c in cases}):
        positions = [(i, j) for i, c in enumerate(cases) if c['dataset'] == dataset for j in range(len(c['examples']))]
        first = min(positions, key=lambda pair: tuple(cases[pair[0]]['examples'][pair[1]]))
        hardest = max(positions, key=lambda pair: float(np.mean(metric_arrays['backward_adaptive']['nmse'][identity[tuple(cases[pair[0]]['examples'][pair[1]])]])))
        chosen.extend([dict(dataset=dataset, selection=label, case=i, position=j) for label, (i, j) in [('fixed_first', first), ('adaptive_hardest', hardest)]])
    selected_cases = {v['case'] for v in chosen}
    waves = {arm: {} for arm in cfg['model']['arms']}
    patch = {arm: dict(nmse=[], rms_gain=[]) for arm in cfg['model']['arms']}
    patch_target_RMS, patch_complete, row_cases = [], [], []
    replayed, usage_review, deployment = [], [], []
    for fold in range(4):
        budget(cfg)
        held = [i for i, c in enumerate(cases) if c['fold'] == fold]
        fit = [i for i, c in enumerate(cases) if c['fold'] != fold]
        for i in held:
            c = cases[i]
            patch_target_RMS.append(np.sqrt(np.mean(c['batch']['x']**2, axis=-1)))
            patch_complete.append(c['complete'])
            row_cases.extend([i]*len(c['examples']))
        for arm in cfg['model']['arms']:
            model = load_model(out/f'{arm}_fold{fold}.npz')
            def encoder_forbidden(*unused, **ignored):
                raise AssertionError('independent replay attempted the encoder')
            model.source.continuous_source = encoder_forbidden
            model.source.analyze = encoder_forbidden
            with np.load(out/f'{arm}_fold{fold}_all_IDs.npz', allow_pickle=False) as stored:
                codes = [{b: stored[f'{b}_{i}'].copy() for b in ('S', 'O')} for i in range(len(cases))]
            with np.load(out/f'{arm}_fold{fold}_held_IDs.npz', allow_pickle=False) as stored:
                for i in held:
                    for b in ('S', 'O'):
                        np.testing.assert_array_equal(stored[f'{b}_{i}'], codes[i][b])
            with np.load(out/f'{arm}_fold{fold}_metrics.npz', allow_pickle=False) as stored:
                retained = {k: stored[k].copy() for k in stored.files}
            cursor = 0
            for i in held:
                budget(cfg)
                c = cases[i]
                ids = codes[i]
                if any(v.dtype.kind not in 'iu' for v in ids.values()):
                    raise TypeError('actual saved integer arrays required')
                hard = model.decode_ids(ids, c['dataset'])
                selected = {k: v[0, c['native']] for k, v in hard.items()}
                audit = joint_reconstruction_metrics(selected, c['batch'], 1, c['scales'])
                count = len(c['examples'])
                np.testing.assert_array_equal(retained['examples'][cursor:cursor+count], c['examples'])
                for k, v in audit['metrics'].items():
                    np.testing.assert_array_equal(retained[k][cursor:cursor+count], v)
                for k, v in audit['support'].items():
                    np.testing.assert_array_equal(retained['support_'+k][cursor:cursor+count], v)
                local_error, local_gain = [], []
                for label, pred in [('x', 'sum'), ('s', 'S'), ('o', 'O')]:
                    y, p, mask = c['batch'][label], selected[pred], c['batch']['mask']
                    denominator = np.where(mask, c['batch']['v'+label], 0).sum(-1)/mask.sum(-1).clip(1)
                    mse = np.where(mask, (p-y)**2, 0).sum(-1)/mask.sum(-1).clip(1)
                    rms_y = np.sqrt(np.where(mask, y*y, 0).sum(-1)/mask.sum(-1).clip(1))
                    rms_p = np.sqrt(np.where(mask, p*p, 0).sum(-1)/mask.sum(-1).clip(1))
                    local_error.append(np.where(c['complete'], mse/np.maximum(denominator, 1e-20), np.nan))
                    local_gain.append(np.where(c['complete'], rms_p/np.maximum(rms_y, 1e-20), np.nan))
                patch[arm]['nmse'].append(np.stack(local_error, axis=-1))
                patch[arm]['rms_gain'].append(np.stack(local_gain, axis=-1))
                if i in selected_cases:
                    waves[arm][i] = selected
                cursor += count
            _, _, errors, support, usage = readout(model, cases, codes, fit, held, backgrounds)
            for b in ('S', 'O'):
                np.testing.assert_array_equal(retained[b+'_pattern_errors'], errors[b])
            with (out/f'{arm}_fold{fold}_code_usage.csv').open() as stream:
                original_usage = list(csv.DictReader(stream))
            for original, recomputed in zip(original_usage, usage, strict=True):
                if any(str(recomputed[k]) != original[k] for k in original):
                    raise AssertionError('independent train/held support count changed')
            usage_review.append(dict(arm=arm, fold=fold, codes=len(usage), support=support))
            all_native = sum(v.size for i in held for v in codes[i].values())
            observed_native = sum(int(cases[i]['observed'].sum())*30 for i in held)
            numeric = sum(v.size for book in model.books.values() for k, v in book.items() if k in ('atoms', 'centers'))
            numeric += sum(v.size for v in model.initial_scales.values())
            deployment.append(dict(arm=arm, fold=fold, vocabulary=513, fixed_width_bits=10,
                integer_count=all_native, observed_native_integer_count=observed_native,
                saved_array_dtype='int32', fixed_width_payload_bits=all_native*10,
                bits_per_second_per_full_native_position_both_branches=10,
                frozen_atom_center_initial_scale_values=numeric, state_per_window_inputs=0,
                model_bytes=(out/f'{arm}_fold{fold}.npz').stat().st_size,
                frozen_source_bytes=(ROOT/cfg['source']['basis_run']/f'fold_model_{fold}.npz').stat().st_size))
            replayed.append(dict(arm=arm, fold=fold, crops=len(held), positions=cursor,
                metric_arrays=len(audit['metrics']), exact_metrics=True, exact_pattern_errors=True,
                encoder_disabled_during_decode=True))
            write_json(out/'review_progress.json', dict(stage='independent_integer_review', fold=fold, arm=arm, CPU_seconds=time.process_time()))
    patch_target_RMS, patch_complete, row_cases = np.concatenate(patch_target_RMS), np.concatenate(patch_complete), np.asarray(row_cases)
    for arm in patch:
        patch[arm] = {k: np.concatenate(v) for k, v in patch[arm].items()}
    np.savez_compressed(out/'patch_startup_audit.npz', examples=examples, input_patch_RMS=patch_target_RMS,
        complete=patch_complete, **{arm+'_'+k: v for arm, data in patch.items() for k, v in data.items()})
    diagnostics = describe_dynamics(cases, patch, patch_target_RMS, patch_complete, row_cases, cfg, out)
    figures = render_review(summary, cases, manifest['rows'], chosen, waves, patch, diagnostics, cfg, out)
    write_json(out/'replay_review.json', dict(status='exact_all_saved_integer_waveform_metric_and_pattern_replay',
        reviews=replayed, support_reviews=usage_review, deployment=deployment, selection=chosen,
        dynamics=diagnostics, figures=figures, prior_CPU_seconds=PRIOR_CPU, CPU_seconds=time.process_time(),
        cumulative_CPU_seconds=PRIOR_CPU+time.process_time(), validation_reads=0, confirmation_reads=0, test_reads=0,
        decoder_input='saved_integer_arrays_and_frozen_model_plus_dataset_native_layout',
        physical_decoder_context='offline_30second_spatial_frequency_synthesis; gain_update_is_backward_on_virtual_atoms'))


def describe_dynamics(cases, patch, input_rms, complete, row_cases, cfg, out):
    """Untrimmed position-patch amplitude bins; the original hierarchy stays in summary."""
    report = dict(amplitude_bin_edges_G0_input_RMS=[0, .5, 1, 2, 4, 'infinity'],
        unit='complete 2-second position patches, dependent within subject', by_dataset={}, state={})
    for dataset in sorted({c['dataset'] for c in cases}):
        dataset_rows = np.asarray([cases[i]['dataset'] == dataset for i in row_cases])
        bins = np.searchsorted([.5, 1., 2., 4.], input_rms, side='right')
        report['by_dataset'][dataset] = {}
        for arm in patch:
            values = []
            for b in range(5):
                selected = complete & dataset_rows[:, None] & (bins == b)
                values.append(dict(bin=b, patches=int(selected.sum()),
                    mean_NMSE=np.mean(patch[arm]['nmse'][selected], axis=0) if selected.any() else [None]*3,
                    median_RMS_gain=np.median(patch[arm]['rms_gain'][selected], axis=0) if selected.any() else [None]*3))
            report['by_dataset'][dataset][arm] = values
    for arm in patch:
        by_branch = {}
        for branch in ('S', 'O'):
            parts = defaultdict(list)
            masks = []
            for fold in range(4):
                with np.load(out/f'{arm}_fold{fold}_state_diagnostics.npz', allow_pickle=False) as stored:
                    for i, c in enumerate(cases):
                        if c['fold'] != fold:
                            continue
                        prefix = f'{branch}_{i}_'
                        masks.append(stored[prefix+'complete'])
                        for k in ('ratio', 'carrier_target_RMS', 'decoded_carrier_RMS'):
                            parts[k].append(stored[prefix+k])
            mask = np.concatenate(masks)
            values = {k: np.concatenate(v) for k, v in parts.items()}
            gain = values['decoded_carrier_RMS']/np.maximum(values['carrier_target_RMS'], 1e-20)
            scale_quantiles, gain_quantiles = [], []
            for p in range(15):
                scale_quantiles.append(np.quantile(values['ratio'][:, p][mask[:, p]], [.1, .5, .9]))
                gain_quantiles.append(np.quantile(gain[:, p][mask[:, p]], [.1, .5, .9]))
            by_branch[branch] = dict(scale_ratio_quantiles=scale_quantiles, virtual_carrier_RMS_gain_quantiles=gain_quantiles,
                first_patch_ratio_exact_one=bool(np.all(values['ratio'][:, 0] == 1)),
                all_scale_ratio_quantiles=np.quantile(values['ratio'][mask], [.001, .1, .5, .9, .999]),
                virtual_RMS_gain_quantiles=np.quantile(gain[mask], [.001, .1, .5, .9, .999]))
        report['state'][arm] = by_branch
    write_json(out/'dynamics_audit.json', report)
    return report


def render_review(summary, cases, rows, chosen, waves, patch, diagnostics, cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.analysis.dual_component_tokenizer import eeg_complete_patch_band_energy
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(fixed='#2685a6', backward_adaptive='#d15e2f')
    names = dict(fixed='Fixed gain', backward_adaptive='Backward adaptive')
    figures = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for n, (arm, data) in enumerate(summary['arms'].items()):
        waveform = data['waveform']['metrics']
        x = np.arange(3)+(-.18 if n == 0 else .18)
        axes[0, 0].bar(x, np.ravel(waveform['nmse']['macro']), .34, color=colors[arm], label=names[arm])
        axes[0, 1].bar(x, np.ravel(waveform['rms_gain']['macro']), .34, color=colors[arm])
        axes[0, 2].bar(x, np.ravel(waveform['local_rms_nrmse']['macro']), .34, color=colors[arm])
        for j, branch in enumerate(('S', 'O')):
            axes[1, j].bar(x, list(data['pattern'][branch]['gain'].values()), .34, color=colors[arm])
        for j, branch in enumerate(('S', 'O')):
            axes[1, 2].bar(j+(-.18 if n == 0 else .18), data['pattern'][branch]['support_mass'], .34, color=colors[arm])
    for ax, title in zip(axes[0], ('Waveform NMSE (lower)', 'Physical RMS gain (target 1)', '0.25-s local RMS NRMSE (lower)')):
        ax.set(xticks=range(3), xticklabels=['X', 'S', 'O'], title=title)
    for j, branch in enumerate(('S', 'O')):
        axes[1, j].set(xticks=range(3), xticklabels=['Theta', 'Alpha', 'Local RMS'], title=f'{branch}: background+single-ID error reduction')
        axes[1, j].axhline(0, color='gray', lw=.7)
    axes[1, 2].set(xticks=[0, 1], xticklabels=['S', 'O'], title='Held patch mass with >=8 fit patches, >=2 people')
    axes[1, 2].axhline(.9, color='black', ls=':', lw=1)
    axes[0, 0].legend()
    fig.suptitle('400 original train crops | 60 subjects | four subject folds | 17,040 matched position-windows\nMacro: positions -> crop -> record -> subject -> dataset; independent entities are subjects')
    path = out/'training_summary.png'
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    figures.append(str(path))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout='constrained')
    for j, branch in enumerate(('S', 'O')):
        for arm in cfg['model']['arms']:
            state = diagnostics['state'][arm][branch]
            for k, field in enumerate(('scale_ratio_quantiles', 'virtual_carrier_RMS_gain_quantiles')):
                q = np.asarray(state[field])
                axes[k, j].plot(np.arange(15)*2, q[:, 1], color=colors[arm], label=names[arm])
                axes[k, j].fill_between(np.arange(15)*2, q[:, 0], q[:, 2], color=colors[arm], alpha=.12)
                axes[k, j].set(yscale='log', ylabel='Ratio (log axis)', xlabel='Patch start in context (s)')
            kcomp = 1 if branch == 'S' else 2
            axes[2, j].plot(np.arange(15)*2, np.nanmean(patch[arm]['nmse'][..., kcomp], axis=0), color=colors[arm], label=names[arm])
        axes[0, j].set_title(branch+': virtual-carrier decoder scale / fixed initial scale')
        axes[1, j].set_title(branch+': decoded / target virtual-carrier RMS')
        axes[2, j].set(title=branch+': original physical waveform mean NMSE by patch', xlabel='Patch start in context (s)', ylabel='NMSE')
        for ax in axes[:, j]:
            ax.grid(alpha=.15)
            ax.legend()
    fig.suptitle('Gain startup and tail: every call resets at 0 s; current atom updates the next scale\nLines are held position-patch medians; shading is 10–90%; bottom is untrimmed dependent patch mean.\nVirtual carrier energy is before the full-30-s spatial frequency synthesis.')
    path = out/'gain_startup_and_tail.png'
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    figures.append(str(path))
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    for selection in chosen:
        c, position = cases[selection['case']], selection['position']
        row = rows[c['crop']['row']]
        scale = c['scales'][position]
        target = {b: c['batch'][b][position].reshape(-1)*scale for b in ('x', 's', 'o')}
        mask = c['batch']['mask'][position].reshape(-1)
        estimates = {arm: {b: waves[arm][selection['case']][k][position].reshape(-1)*scale for b, k in [('x', 'sum'), ('s', 'S'), ('o', 'O')]} for arm in cfg['model']['arms']}
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), layout='constrained')
        time_grid = np.arange(6000)/200
        for r, label in enumerate(('x', 's', 'o')):
            values = np.concatenate([target[label][mask]]+[estimates[arm][label] for arm in estimates])
            lo, hi = values.min(), values.max()
            pad = max((hi-lo)*.05, 1e-8)
            for column, span in enumerate(((0, 30), (14, 16), (13.5, 14.5))):
                ax = axes[r, column]
                ax.plot(time_grid, np.where(mask, target[label], np.nan), color='#20384d', lw=.8, label='Target')
                for arm in estimates:
                    ax.plot(time_grid, estimates[arm][label], color=colors[arm], lw=.7, alpha=.9, label=names[arm])
                for edge in np.arange(2, 30, 2):
                    ax.axvline(edge, color='gray', ls=':', lw=.5)
                edges = np.diff(np.r_[False, ~mask, False].astype(int))
                for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    ax.axvspan(start/200, stop/200, color='gray', alpha=.2)
                ax.set(xlim=span, ylim=(lo-pad, hi+pad), xlabel='Crop time (s)')
                if column == 0:
                    ax.set_ylabel(label.upper()+' | '+row.get('source_unit', 'source unit'))
                if r == 0:
                    ax.set_title(('Full 30 s', 'Fixed 14–16 s', 'Fixed 14-s boundary')[column])
                ax.grid(alpha=.1)
        axes[0, 0].legend(loc='lower left', fontsize=8)
        heading = f"{c['dataset']} | {c['subject']} | {row['record']} | {row['positions'][int(c['examples'][position, 2])]}\nrow {c['crop']['row']}, start {c['crop']['start_s']} s | {selection['selection']} | no waveform smoothing"
        fig.suptitle(heading)
        fig.supxlabel('Same target/prediction coordinates and full-window y limits in all columns. Dots: 2-s boundaries; gray: unscored mask.')
        path = out/f"waveform_{c['dataset']}_{selection['selection']}.png"
        fig.savefig(path, dpi=200, facecolor='white')
        plt.close(fig)
        figures.append(str(path))
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
        patch_mask = mask.reshape(1, 15, 400, 1)
        for r, label in enumerate(('x', 's', 'o')):
            signals = [('Target', '#20384d', target[label])]+[(names[arm], colors[arm], estimates[arm][label]) for arm in estimates]
            for k, (name, color, signal) in enumerate(signals):
                rms = np.sqrt(np.mean(signal.reshape(120, 50)**2, axis=1))
                ok = mask.reshape(120, 50).all(1)
                axes[r, 0].plot((np.arange(120)+.5)/4, np.where(ok, rms, np.nan), color=color, label=name)
                energy = eeg_complete_patch_band_energy(signal.reshape(1, 15, 400, 1), patch_mask)
                axes[r, 1].plot(range(5), [energy[band][0, 0] for band in bands], marker='o', color=color, label=name)
            axes[r, 0].set(xlim=(0, 30), ylabel=label.upper()+' RMS | '+row.get('source_unit', 'source unit'), xlabel='Crop time (s)')
            axes[r, 1].set(yscale='log', xticks=range(5), xticklabels=['1–4', '4–8', '8–13', '13–30', '30–45'], xlabel='Band (Hz)', ylabel='Hann energy | source unit squared')
        axes[0, 0].legend()
        fig.suptitle(heading+'\n0.25-s original-voltage RMS and complete 2-s band energies; all valid support retained')
        path = out/f"dynamics_{c['dataset']}_{selection['selection']}.png"
        fig.savefig(path, dpi=200, facecolor='white')
        plt.close(fig)
        figures.append(str(path))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', action='store_true', help='independent saved-integer replay and fixed diagnostic figures')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    with threadpool_limits(limits=cfg['resources']['BLAS_threads_per_book'], user_api='blas'):
        try:
            if args.review:
                review(args, cfg)
            else:
                run(args, cfg)
        except Exception as error:
            write_json(args.output/('review_failure.json' if args.review else 'failure.json'), dict(error=repr(error), CPU_seconds=time.process_time()))
            raise


if __name__ == '__main__':
    main()
