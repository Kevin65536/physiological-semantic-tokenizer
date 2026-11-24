#!/usr/bin/env python3
"""Independently replay the frozen spatial reference on every development crop."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
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

from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics, joint_reconstruction_summary
from src.analysis.spatial_tokenizer import edge_metrics
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.spatial_waveform_dictionary import load_model
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from src.tokenizers.spatial_local_gain import load_decoder

STATE = {}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, default=encode) + '\n')
    temporary.replace(path)


def save(path, **values):
    temporary = path.with_suffix('.pending.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def load_checkpoint(path):
    return load_decoder(path) if STATE.get('as_encoder') else load_model(path)


def load_scope(args):
    cfg = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    panel = json.loads(args.panels.with_name('panel.json').read_text())
    if (panel['panel_sha256'] != digest(args.panels) or panel['cfg_identity'] != manifest['cfg_identity']
            or panel['scales_identity'] != manifest['scales_identity']):
        raise ValueError('registered G0 panel identity changed before signal arrays')
    rows = manifest['rows']
    with np.load(args.panels, allow_pickle=False) as stored:
        examples = {m: stored[m+'_development'].copy() for m in ('eeg', 'hb')}
        confirms = {m: set(map(tuple, stored[m+'_confirm'])) for m in ('eeg', 'hb')}
    for modality, values in examples.items():
        for index, _, _ in values:
            row = rows[index]
            if (row['split'] != 'validation' or row['modality'] != modality
                    or row['subject'] == panel['confirmation_subjects'][row['dataset']]):
                raise ValueError('development-only subject boundary before arrays')
        if set(map(tuple, values)) & confirms[modality]:
            raise ValueError('confirmation overlap before arrays')
    as_paths = (args.as_encoder, args.screen_ids, args.screen_metrics)
    if any(path is not None for path in as_paths) and not all(path is not None for path in as_paths):
        raise ValueError('AS audit requires encoder, frozen screen IDs and metrics together')
    STATE['as_encoder'] = args.as_encoder
    model = load_checkpoint(args.checkpoint)
    if (model.metadata['input_identity'] != manifest['cfg_identity']
            or model.metadata['scales_identity'] != manifest['scales_identity']):
        raise ValueError('frozen spatial G0 coordinate mismatch')
    grouped = defaultdict(list)
    for i, start, position in examples['eeg']:
        grouped[rows[i]['dataset'], int(i), int(start)].append(int(position))
    datasets = sorted(model.metadata['positions'])
    jobs = {d: [(i, start, positions) for (dataset, i, start), positions in grouped.items() if dataset == d]
            for d in datasets}
    signature = dict(panel_sha256=panel['panel_sha256'], checkpoint_sha256=digest(args.checkpoint),
                     cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'])
    encoder = model
    reused = {}
    if args.as_encoder:
        encoder = load_encoder(args.as_encoder)
        source = json.loads(args.as_encoder.read_text())
        if source['source_sha256'] != model.metadata['source_decoder_sha256']:
            raise ValueError('AS search and local-gain decoder must share frozen source')
        if encoder.metadata['positions'] != model.metadata['positions']:
            raise ValueError('AS encoder and decoder native position identity mismatch')
        metadata_path = args.screen_ids.with_name('integer_metadata.json')
        metadata = json.loads(metadata_path.read_text())
        with np.load(args.panels, allow_pickle=False) as stored:
            screen_crops = {(int(i), int(start)) for i, start, _ in stored['eeg_screen']}
        if {(int(v['row']), int(v['start_s'])) for v in metadata} != screen_crops:
            raise ValueError('reuse IDs must cover exactly original frozen screen crops')
        allowed = {(i, start) for group in jobs.values() for i, start, _ in group}
        with np.load(args.screen_ids, allow_pickle=False) as stored:
            for item in metadata:
                key = int(item['row']), int(item['start_s'])
                dataset = rows[key[0]]['dataset']
                if key not in allowed or item['native_positions'] != model.metadata['positions'][dataset]:
                    raise ValueError('retained AS IDs differ from authorized native development crop')
                reused[key] = {b: stored[f'{b}_{item["crop"]}'].copy() for b in ('S', 'O')}
        signature.update(as_encoder_sha256=digest(args.as_encoder),
                         screen_ids_sha256=digest(args.screen_ids), screen_metadata_sha256=digest(metadata_path),
                         screen_metrics_sha256=digest(args.screen_metrics), source_decoder_sha256=source['source_sha256'])
    STATE.update(rows=rows, scales=scales, examples=examples, model=model, encoder=encoder, checkpoint=args.checkpoint,
                 output=args.output, jobs=jobs, signature=signature, threads=2, reused=reused,
                 reference=args.reference_root or args.output, screen_metrics=args.screen_metrics)
    plan = dict(signature=signature, role='development', confirmation_excluded=panel['confirmation_subjects'],
                counts={d: dict(crops=len(jobs[d]), positions=sum(len(c[2]) for c in jobs[d])) for d in datasets},
                grouping='same parent row and start jointly encoded once across native positions',
                source_panel=str(args.panels), checkpoint=str(args.checkpoint), fitting=False)
    if args.as_encoder:
        plan.update(as_encoder=str(args.as_encoder), reused_screen_crops=len(reused), new_AS_search_crops=sum(map(len, jobs.values()))-len(reused),
                    screen_metrics=str(args.screen_metrics), reference_root=str(STATE['reference']),
                    decode_passes_from_saved_disk_integers=2, rule='frozen original full-array AS search and frozen train-CV local gain; no further tuning')
    path = args.output/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('retained development audit plan changed')
    write(path, plan)
    return plan


def historical_denominators():
    root = STATE['output']/'historical_primary'
    root.mkdir(exist_ok=True)
    if (root/'verification.json').exists():
        return
    rows, scales = STATE['rows'], STATE['scales']
    evidence = []
    legacy = ROOT/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261122_unified_base_g1_v1'
    for modality in ('eeg', 'hb'):
        examples = STATE['examples'][modality]
        colors = 1 if modality == 'eeg' else 2
        denominator = np.array([[scales['positions'][modality][rows[i]['dataset']][rows[i]['positions'][p]]['target_variance'][b]
                                 for b in ('x', 's', 'o')] for i, _, p in examples], dtype=np.float32).reshape(len(examples), 3, colors)
        for candidate in ('vq128', 'continuous'):
            source = legacy/'models'/f'{modality}__waveform__{candidate}'/'evaluation/step_10000/validation.npz'
            with np.load(source, allow_pickle=False) as stored:
                lookup = {tuple(v): n for n, v in enumerate(stored['examples'])}
                chosen = np.array([lookup[tuple(v)] for v in examples])
                fields = ['nmse', 'nrmse', 'rmse', 'bias', 'correlation', 'rms_gain', 'level_nmse', 'shape_nmse']
                fields += [k for k in stored.files if k.startswith('band_')]
                metrics = {k: stored[k][chosen] for k in fields}
                recomputed = metrics['rmse']**2/denominator
                maximum = float(np.max(abs(recomputed-metrics['nmse'])))
                np.testing.assert_allclose(recomputed, metrics['nmse'], rtol=1e-12, atol=1e-12)
            summary = base_training_aggregate(metrics, examples, rows)
            summary.update(source=str(source), role='development', modality=modality,
                denominator='historical float32 G0 train component/color variance in input-SD coordinate',
                rms_gain='retained historical implementation; exact-zero target corrections are evaluated by new CLI',
                max_abs_denominator_recalculation_difference=maximum)
            save(root/f'{modality}_{candidate}_metrics.npz', examples=examples, **metrics)
            write(root/f'{modality}_{candidate}_summary.json', summary)
            evidence.append(dict(modality=modality, candidate=candidate, examples=len(examples), max_difference=maximum))
    write(root/'verification.json', dict(status='passed', comparisons=evidence,
        support='all exact frozen development triples; confirmation excluded from every aggregate'))


def evaluate_crop(model, dataset, crop, cache, ids=None):
    rows, scales = STATE['rows'], STATE['scales']
    index, start, positions = crop
    row = rows[index]
    names = model.metadata['positions'][dataset]
    if ids is None:
        ids = STATE.get('reused', {}).get((index, start))
    if ids is None:
        fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(row=index, start_s=start, positions=positions),
            scales, names, cache, allowed_splits=('validation',))
        values = fields['x'].T.reshape(1, len(names), 15, 400)
        ids = STATE['encoder'].encode(values, dataset, observed[None])
    prediction = model.decode_ids(ids, dataset)
    examples = [(index, start, p) for p in positions]
    batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=8)
    mapping = {name.upper(): j for j, name in enumerate(names)}
    selected = [mapping[row['positions'][p].upper()] for p in positions]
    prediction = {b: prediction[b][0, selected] for b in ('S', 'O', 'sum')}
    factors = np.array([scales['positions']['eeg'][dataset][row['positions'][p]]['scale'] for p in positions])
    result = joint_reconstruction_metrics(prediction, batch, 1, factors)
    result['metrics'].update(edge_metrics(prediction, batch))
    return result, ids, examples


def benchmark():
    path = STATE['output']/'benchmark.json'
    if STATE.get('as_encoder') and path.exists():
        record = json.loads(path.read_text())
        STATE['threads'] = record['selected_threads']
        return record
    results = []
    settings = (1, 2) if STATE.get('as_encoder') else (1, 2, 4)
    for threads in settings:
        with threadpool_limits(threads):
            for dataset, crops in STATE['jobs'].items():
                cache, count = OrderedDict(), 0
                selected = [crop for crop in crops if (crop[0], crop[1]) not in STATE.get('reused', {})][:4] if STATE.get('as_encoder') else crops[:8]
                started = time.monotonic()
                for crop in selected:
                    _, _, examples = evaluate_crop(STATE['model'], dataset, crop, cache)
                    count += len(examples)
                duration = time.monotonic()-started
                results.append(dict(threads=threads, dataset=dataset, crops=len(selected), examples=count,
                                    seconds=duration, examples_per_second=count/duration))
    totals = {t: sum(v['seconds'] for v in results if v['threads'] == t) for t in settings}
    chosen = min(totals, key=totals.get)
    STATE['threads'] = chosen
    estimates = {d: next(v for v in results if v['dataset'] == d and v['threads'] == chosen)['seconds']/next(v for v in results if v['dataset'] == d and v['threads'] == chosen)['crops']*len(crops)*(3 if STATE.get('as_encoder') else 2)
                 for d, crops in STATE['jobs'].items()}
    record = dict(status='completed', results=results, selected_threads=chosen, workers=4,
        estimated_two_pass_seconds_by_dataset=estimates, estimated_wall_seconds=max(estimates.values()),
        reason='four independent dataset workers share immutable metadata and use fastest measured thread setting')
    write(STATE['output']/'benchmark.json', record)
    return record


def worker(payload):
    dataset, phase = payload
    out = STATE['output']/'spatial'/dataset
    out.mkdir(exist_ok=True, parents=True)
    crops = STATE['jobs'][dataset]
    model = load_checkpoint(STATE['checkpoint']) if phase != 'forward' else STATE['model']
    cache = OrderedDict()
    started = time.monotonic()
    with threadpool_limits(STATE['threads']):
        for block_start in range(0, len(crops), 32):
            block = out/f'{phase}_{block_start//32:04d}.npz'
            if block.exists():
                with np.load(block, allow_pickle=False) as stored:
                    if json.loads(str(stored['signature'])) != STATE['signature']:
                        raise ValueError('retained block identity changed')
                continue
            metrics, support, codes = defaultdict(list), defaultdict(list), defaultdict(list)
            examples = []
            input_block = None
            if phase != 'forward':
                input_block = np.load(out/f'forward_{block_start//32:04d}.npz', allow_pickle=False)
            try:
                for local, crop in enumerate(crops[block_start:block_start+32]):
                    ids = None if input_block is None else {b: input_block['ids_'+b][local:local+1] for b in ('S', 'O')}
                    audited, ids, take = evaluate_crop(model, dataset, crop, cache, ids)
                    examples.extend(take)
                    for key, value in audited['metrics'].items():
                        metrics[key].append(value)
                    for key, value in audited['support'].items():
                        support[key].append(value)
                    for b, value in ids.items():
                        codes[b].append(value[0])
                arrays = {key: np.concatenate(value) for key, value in metrics.items()}
                arrays.update({'support_'+key: np.concatenate(value) for key, value in support.items()})
                if input_block is not None:
                    for key, value in arrays.items():
                        np.testing.assert_allclose(value, input_block[key], rtol=0, atol=0, equal_nan=True, err_msg=key)
                save(block, signature=np.asarray(json.dumps(STATE['signature'], sort_keys=True)),
                    examples=np.asarray(examples), crop_row_start=np.asarray([(i, s) for i, s, _ in crops[block_start:block_start+32]]),
                    **arrays, **{'ids_'+b: np.stack(value) for b, value in codes.items()})
            finally:
                if input_block is not None:
                    input_block.close()
            write(out/'progress.json', dict(phase=phase, crops=min(block_start+32, len(crops)), total_crops=len(crops),
                elapsed_seconds=time.monotonic()-started, pid=os.getpid()))
    write(out/(phase+'_completion.json'), dict(status='completed', pid=os.getpid(),
        crops=len(crops), elapsed_seconds=time.monotonic()-started, encoder_called=phase == 'forward'))
    return dict(dataset=dataset, phase=phase, elapsed_seconds=time.monotonic()-started)


def merge():
    collections = defaultdict(list)
    examples = []
    for dataset in STATE['jobs']:
        out = STATE['output']/'spatial'/dataset
        for path in sorted(out.glob('replay_*.npz')):
            with np.load(path, allow_pickle=False) as stored:
                examples.append(stored['examples'])
                for key in stored.files:
                    if key not in ('examples', 'signature', 'crop_row_start', 'ids_S', 'ids_O'):
                        collections[key].append(stored[key])
    examples = np.concatenate(examples)
    lookup = {tuple(value): n for n, value in enumerate(examples)}
    frozen = STATE['examples']['eeg']
    if len(examples) != len(frozen) or len(lookup) != len(frozen):
        raise ValueError('complete exact development denominator required')
    order = np.array([lookup[tuple(value)] for value in frozen])
    values = {key: np.concatenate(parts)[order] for key, parts in collections.items()}
    metrics = {k: v for k, v in values.items() if not k.startswith('support_')}
    support = {k.removeprefix('support_'): v for k, v in values.items() if k.startswith('support_')}
    save(STATE['output']/'spatial/metrics.npz', examples=frozen, **values)
    summary = joint_reconstruction_summary(metrics, support, frozen.tolist(), STATE['rows'])
    summary.update(status='completed', modality='eeg', role='development', checkpoint=str(STATE['checkpoint']),
        checkpoint_sha256=STATE['signature']['checkpoint_sha256'], fitting=False,
        integer_replay=dict(status='passed', source='per-dataset forward_*.npz integer arrays',
            independent_processes=True, metrics_max_abs_difference=0., encoder_called_in_replay=False,
            disk_decode_passes=2 if STATE.get('as_encoder') else 1),
        panel_sha256=STATE['signature']['panel_sha256'])
    write(STATE['output']/'spatial/summary.json', summary)
    return summary


def export_examples():
    """Use fixed identities and baseline-defined tails, then decode their saved IDs."""
    out = STATE['output']/'spatial'
    target = out/'display_v1'
    target.mkdir(exist_ok=True)
    if (target/'waveform_examples.npz').exists():
        raise FileExistsError('retained development waveform export already exists')
    with np.load(out/'metrics.npz', allow_pickle=False) as stored:
        examples, current = stored['examples'].copy(), stored['nmse'].mean((1, 2))
    with np.load(STATE['reference']/'eeg/vq128/metrics.npz', allow_pickle=False) as stored:
        np.testing.assert_array_equal(examples, stored['examples'])
        reference = stored['nmse'].mean((1, 2))
    rows, scales = STATE['rows'], STATE['scales']
    selection = defaultdict(list)
    for dataset in STATE['jobs']:
        group = [n for n, (i, _, _) in enumerate(examples) if rows[i]['dataset'] == dataset]
        group.sort(key=lambda n: tuple(examples[n]))
        selection[group[0]].append('fixed_identity_first')
        selection[group[int(np.random.default_rng(23017).integers(len(group)))]].append('fixed_random_seed23017')
        finite = [n for n in group if np.isfinite(reference[n])]
        percentile = np.quantile(reference[finite], .9)
        selection[min(finite, key=lambda n: (abs(reference[n]-percentile), n))].append('reference_nearest_p90')
        selection[max(finite, key=lambda n: (reference[n], -n))].append('reference_max')
        finite = [n for n in group if np.isfinite(current[n])]
        selection[max(finite, key=lambda n: (current[n], -n))].append('candidate_max')
        for n in group:
            if not np.isfinite(current[n]):
                selection[n].append('candidate_failed')
    indices = sorted(selection)
    take = examples[indices]
    batch = base_tokenizer_batch(rows, take, scales)
    arrays = {key: batch[key] for key in ('x', 's', 'o', 'mask')}
    predictions = {b: [] for b in ('S', 'O', 'sum')}
    crop_lookup = {(i, start): (dataset, j) for dataset, crops in STATE['jobs'].items()
                   for j, (i, start, _) in enumerate(crops)}
    model = load_checkpoint(STATE['checkpoint'])
    metadata = []
    for ordinal, (index, start, p) in zip(indices, take):
        row = rows[index]
        dataset, crop = crop_lookup[int(index), int(start)]
        block = out/dataset/f'forward_{crop//32:04d}.npz'
        with np.load(block, allow_pickle=False) as stored:
            ids = {b: stored['ids_'+b][crop % 32:crop % 32+1] for b in ('S', 'O')}
        decoded = model.decode_ids(ids, dataset)
        names = model.metadata['positions'][dataset]
        native = next(j for j, name in enumerate(names) if name.upper() == row['positions'][p].upper())
        for b in predictions:
            predictions[b].append(decoded[b][0, native])
        metadata.append(dict(modality='eeg', dataset=dataset, subject=row['subject'], record=row['record'],
            parent=row['parent'], row=int(index), start_s=int(start), position=int(p), position_name=row['positions'][p],
            scale=float(scales['positions']['eeg'][dataset][row['positions'][p]]['scale']), source_unit=row['source_unit'],
            panel_index=ordinal, selection=selection[ordinal], token_seconds=2,
            prediction_label='Frozen AS local-energy integer decode' if STATE.get('as_encoder') else 'Spatial integer decode',
            interface='one_ID_per_2s_per_branch_position_full_native_EEG_integer_sequences',
            decoder_input=str(block), decoded_native_positions=len(names),
            decode_condition='both branches, every native position, saved integer sequences and frozen model only'))
    arrays.update({'prediction_'+b: np.stack(values) for b, values in predictions.items()})
    save(target/'waveform_examples.npz', **arrays)
    write(target/'waveform_examples.json', dict(coordinate='G0_normalized', model=str(STATE['checkpoint']),
        role='development', reference_metrics=str(STATE['reference']/'eeg/vq128/metrics.npz'),
        source_examples=len(examples), examples=metadata, encoder_called=False))



def input_support_worker(dataset):
    """Inspect unchanged X/target support and FFT tails without refitting or rescoring."""
    examples, fractions, peaks, coverage = [], [], [], []
    counts = defaultdict(int)
    previous, maps = None, {}
    started = time.monotonic()
    with threadpool_limits(1):
        for index, start, positions in STATE['jobs'][dataset]:
            row = STATE['rows'][index]
            if row['folder'] != previous:
                maps = {k: np.load(Path(row['folder'])/(k+'.npy'), mmap_mode='r')
                        for k in ('x', 's', 'o', 'mask', 'observation_valid')}
                previous = row['folder']
            interval = slice(start*200, (start+30)*200)
            x = np.asarray(maps['x'][interval, :, 0], float)
            observed = np.asarray(maps['observation_valid'][interval, :, 0], bool)
            finite = np.isfinite(x)
            complete = observed.all(0) & finite.all(0)
            counts['native_position_crops'] += x.shape[1]
            counts['native_complete_observed_position_crops'] += int(complete.sum())
            counts['native_nonfinite_x_points'] += int((~finite).sum())
            counts['native_observed_nonfinite_x_points'] += int((observed & ~finite).sum())
            selected = np.asarray(positions)
            counts['planned_position_crops'] += len(selected)
            counts['planned_incomplete_observed_position_crops'] += int((~complete[selected]).sum())
            mask = np.asarray(maps['mask'][interval, selected, 0], bool)
            counts['planned_scored_points'] += int(mask.sum())
            for key in ('x', 's', 'o'):
                target = np.asarray(maps[key][interval, selected, 0], float)
                counts[key+'_nonfinite_on_scoring_support'] += int((mask & ~np.isfinite(target)).sum())
            values = x[:, selected]
            fft_valid = complete[selected]
            centered = np.where(fft_valid[None], values, 0.)
            centered = centered-centered.mean(0)
            power = abs(np.fft.rfft(centered, axis=0))**2
            frequency = np.fft.rfftfreq(6000, 1/200)
            # Double interior real-FFT bins to conserve full real-signal energy.
            power[1:-1] *= 2
            denominator = power.sum(0)
            bands = np.stack([power[frequency < 1].sum(0), power[frequency >= 45].sum(0)])
            ratio = np.divide(bands, denominator[None], out=np.full(bands.shape, np.nan), where=denominator[None] > 0).T
            ratio[~fft_valid] = np.nan
            peak = frequency[np.argmax(power, axis=0)]
            peak[~fft_valid | (denominator == 0)] = np.nan
            examples.extend((index, start, int(position)) for position in positions)
            fractions.extend(ratio)
            peaks.extend(peak)
            coverage.extend(fft_valid)
    out = STATE['output']/'spatial'/dataset
    save(out/'input_support_v1.npz', examples=np.asarray(examples),
         full_30s_observed=np.asarray(coverage), fft_energy_fraction=np.asarray(fractions),
         dominant_hz=np.asarray(peaks))
    write(out/'input_support_v1.json', dict(status='completed', dataset=dataset, counts=dict(counts),
        pid=os.getpid(), seconds=time.monotonic()-started,
        diagnostic='unchanged full 30-s X, demeaned real FFT; fractions [f<1Hz, f>=45Hz], no score or mask changes'))
    return dict(dataset=dataset, counts=dict(counts), seconds=time.monotonic()-started)


def input_support():
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        result = list(pool.map(input_support_worker, STATE['jobs']))
    write(STATE['output']/'spatial/input_support_v1.json', dict(status='completed', datasets=result,
        signal_scope='same frozen development crop support; no confirmation/test reads',
        bandpass_owner='src/data/unified_physiology.py:_bandpass uses fourth-order Butterworth sosfiltfilt; 45Hz is cutoff, not zero stop-band',
        interpretation='retained >45Hz voltage is scored unchanged; FFT fractions are diagnostics, not a new gate'))


def verify_original_screen_packing():
    """Keep exact replay checking while separately recording batch-layout roundoff."""
    path = STATE['output']/'spatial/screen_replay_v1.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['signature'] != STATE['signature']:
            raise ValueError('retained exact screen replay identity changed')
        return previous
    model = load_checkpoint(STATE['checkpoint'])
    crop_lookup = {(i, start): (dataset, j, positions) for dataset, crops in STATE['jobs'].items()
                   for j, (i, start, positions) in enumerate(crops)}
    metrics, support, ordering = defaultdict(list), defaultdict(list), []
    groups = defaultdict(list)
    with np.load(STATE['screen_metrics'], allow_pickle=False) as stored:
        expected = {k: stored[k].copy() for k in stored.files}
    for ordinal, (i, start, p) in enumerate(expected['examples']):
        groups[int(i), int(start)].append((ordinal, int(p)))
    probes, cache = [], OrderedDict()
    for (i, start), selected in groups.items():
        dataset, crop, full_positions = crop_lookup[i, start]
        block = STATE['output']/'spatial'/dataset/f'forward_{crop//32:04d}.npz'
        with np.load(block, allow_pickle=False) as stored:
            ids = {b: stored['ids_'+b][crop%32:crop%32+1] for b in ('S', 'O')}
        for b in ('S', 'O'):
            np.testing.assert_array_equal(ids[b], STATE['reused'][i, start][b])
        positions = [p for _, p in selected]
        audited, _, _ = evaluate_crop(model, dataset, (i, start, positions), cache, ids)
        if dataset not in {v['dataset'] for v in probes}:
            full_batch = base_tokenizer_batch(STATE['rows'], [(i, start, p) for p in full_positions], STATE['scales'], array_cache=cache)
            small_batch = base_tokenizer_batch(STATE['rows'], [(i, start, p) for p in positions], STATE['scales'], array_cache=cache)
            chosen = [full_positions.index(p) for p in positions]
            fields = []
            for key in full_batch:
                if isinstance(full_batch[key], np.ndarray) and full_batch[key].shape[0] == len(full_positions):
                    np.testing.assert_array_equal(full_batch[key][chosen], small_batch[key])
                    fields.append(key)
            probes.append(dict(dataset=dataset, row=i, start_s=start, full_positions=len(full_positions), screen_positions=len(positions),
                identical_target_mask_variance_fields=fields, wave_input='same saved full-field integer IDs and frozen decoder',
                only_changed_argument='number of selected positions presented to metric owner'))
        ordering.extend(n for n, _ in selected)
        for key, value in audited['metrics'].items():
            if key in expected:
                metrics[key].append(value)
        for key, value in audited['support'].items():
            support['support_'+key].append(value)
    order = np.argsort(ordering)
    arrays = {k: np.concatenate(v)[order] for k, v in dict(metrics, **support).items()}
    for key, value in arrays.items():
        np.testing.assert_allclose(value, expected[key], rtol=0, atol=0, equal_nan=True, err_msg=key)
    save(path.with_suffix('.npz'), examples=expected['examples'], **arrays)
    result = dict(status='passed', signature=STATE['signature'], inputs=len(order), crops=len(groups),
                  rtol=0, atol=0, all_metric_support_max_abs_difference=0.,
                  same_full_development_saved_IDs=True, encoder_called=False, packing_probes=probes)
    write(path, result)
    return result


def review():
    """Summarize retained numerical artifacts and subject-level comparisons."""
    out = STATE['output']/'spatial'
    summaries = {name: json.loads((path/'summary.json').read_text()) for name, path in
        [('spatial', out), ('vq128', STATE['reference']/'eeg/vq128'), ('continuous', STATE['reference']/'eeg/continuous')]}
    with np.load(out/'metrics.npz', allow_pickle=False) as source:
        examples = source['examples'].copy()
        values = {k: source[k].copy() for k in source.files if k != 'examples'}
    for name in ('vq128', 'continuous'):
        with np.load(STATE['reference']/f'eeg/{name}/metrics.npz', allow_pickle=False) as source:
            np.testing.assert_array_equal(examples, source['examples'])
    screen_path = STATE['screen_metrics'] or STATE['checkpoint'].parent/'spatial_metrics.npz'
    lookup = {tuple(value): n for n, value in enumerate(examples)}
    with np.load(screen_path, allow_pickle=False) as source:
        chosen = [lookup[tuple(value)] for value in source['examples']]
        difference = float(np.max(abs(values['nmse'][chosen]-source['nmse'])))
        reproduced = {}
        for key in source.files:
            if key != 'examples':
                actual, expected = values[key][chosen], source[key]
                reproduced[key] = dict(bitwise_equal=bool(np.array_equal(actual, expected, equal_nan=True)),
                    max_abs_difference=int(np.sum(actual != expected)) if actual.dtype == bool else float(np.nanmax(abs(actual-expected))))
        if STATE.get('as_encoder'):
            for key in ('nmse', 'nrmse', 'rmse', 'rmse_native', 'bias', 'bias_native', 'correlation', 'rms_gain'):
                np.testing.assert_allclose(values[key][chosen], source[key], rtol=0, atol=0, equal_nan=True, err_msg=key)
    exact_screen = verify_original_screen_packing() if STATE.get('as_encoder') else None
    from experiments.scripts.review_joint_tokenizer import score
    contract = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml').read_text())
    decision = score(summaries['spatial'], summaries['vq128'], summaries['continuous'], contract)
    write(out/'reconstruction_decision_v1.json', decision)
    if STATE.get('as_encoder'):
        summaries['old_spatial'] = json.loads((STATE['reference']/'spatial/summary.json').read_text())

    ranks = {}
    for dataset in STATE['jobs']:
        group = np.asarray([n for n, (i, _, _) in enumerate(examples) if STATE['rows'][i]['dataset'] == dataset])
        metrics = {}
        for key in ('nmse', 'rmse_native', 'bias_native', 'correlation', 'rms_gain', 'band_4_8_abs_db', 'band_8_13_abs_db'):
            a = values[key][group]
            metrics[key] = dict(quantiles_0_50_90_95_100=np.nanquantile(a, [0, .5, .9, .95, 1], axis=0),
                                nonfinite=int((~np.isfinite(a)).sum()))
        ranks[dataset] = dict(examples=len(group), subjects=sorted({STATE['rows'][examples[n, 0]]['subject'] for n in group}), metrics=metrics)
    subject_indexes = {name: {(v['dataset'], v['subject']): np.asarray(v['value']).reshape(3)
                       for v in summary['metrics']['nmse']['by_subject']}
                       for name, summary in summaries.items()}
    subjects = []
    for identity, current in subject_indexes['spatial'].items():
        subjects.append(dict(dataset=identity[0], subject=identity[1], nmse=current,
            k128_minus_spatial=subject_indexes['vq128'][identity]-current,
            continuous_minus_spatial=subject_indexes['continuous'][identity]-current))
    if 'old_spatial' in subject_indexes:
        for item in subjects:
            identity = item['dataset'], item['subject']
            item['old_spatial_minus_current'] = subject_indexes['old_spatial'][identity] - np.asarray(item['nmse'])
    support = {}
    for dataset in STATE['jobs']:
        path = out/dataset/'input_support_v1.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as source:
                fractions = source['fft_energy_fraction'][:, 1]
                order = np.argsort(np.where(np.isfinite(fractions), fractions, -1))[-5:][::-1]
                support[dataset] = dict(full_30s_fft_examples=len(fractions),
                    nonfinite_fractions=int((~np.isfinite(fractions)).sum()),
                    ge45hz_fraction_p50_p90_p100=np.nanquantile(fractions, [.5, .9, 1]),
                    ge45hz_fraction_greater_point1_count=int((fractions > .1).sum()),
                    largest=[dict(example=source['examples'][n], fraction=fractions[n], dominant_hz=source['dominant_hz'][n])
                             for n in order])
    writes = dict(status='completed', frozen_signature=STATE['signature'], input_positions=len(examples),
        independent_subjects=summaries['spatial']['subjects'], physical_windows=summaries['spatial']['windows'],
        confirmation_excluded=json.loads((STATE['output']/'plan.json').read_text())['confirmation_excluded'],
        matched_panel_macro={name: summary['selection_score'] for name, summary in summaries.items()},
        matched_panel_x_s_o_nmse={name: summary['metrics']['nmse']['macro'] for name, summary in summaries.items()},
        screen_reproduction=dict(count=len(chosen), max_abs_nmse_difference=difference, full_development_batch_extract=reproduced, exact_original_screen_packing=exact_screen,
            explanation='same saved frozen model and identities exactly reproduce screen; population expansion changes the aggregate'),
        by_dataset_distribution=ranks, by_subject_comparison=subjects,
        subject_improvement_counts_x_s_o=np.sum([np.asarray(item['k128_minus_spatial']) > 0 for item in subjects], axis=0),
        fft_outside_band_diagnostic=support,
        conclusion=dict(reconstruction_passed=decision['reconstruction_passed'], failed_checks=decision['failed']),
        independent_units='16 subjects nested in 4 datasets; position windows are dependent observations',
        decision=str(out/'reconstruction_decision_v1.json'),
        input_support=str(out/'input_support_v1.json'), visual_audit=str(out/'visual_audit_v1.json'))
    write(out/'independent_review_v1.json', writes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--as-encoder', type=Path)
    parser.add_argument('--screen-ids', type=Path)
    parser.add_argument('--screen-metrics', type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--support-only', action='store_true')
    parser.add_argument('--review-only', action='store_true')
    args = parser.parse_args()
    args.panels, args.checkpoint, args.output = (path.resolve() for path in (args.panels, args.checkpoint, args.output))
    for name in ('as_encoder', 'screen_ids', 'screen_metrics', 'reference_root'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(exist_ok=True, parents=True)
    load_scope(args)
    if args.support_only:
        input_support()
        return
    if args.review_only:
        review()
        return
    if args.export_only:
        export_examples()
        return
    if not args.as_encoder:
        historical_denominators()
    record = benchmark()
    if args.benchmark_only:
        return
    for phase in (('forward', 'replay', 'replay2') if args.as_encoder else ('forward', 'replay')):
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
            results = list(pool.map(worker, [(d, phase) for d in STATE['jobs']]))
        write(args.output/(phase+'_completion.json'), dict(status='completed', results=results))
    summary = merge()
    if args.as_encoder:
        review()
        export_examples()
    print(json.dumps(dict(status='completed', inputs=summary['inputs'], macro_NMSE=summary['selection_score'])))


if __name__ == '__main__':
    main()
