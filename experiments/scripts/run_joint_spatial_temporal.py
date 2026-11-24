#!/usr/bin/env python3
"""Fit complementary EEG time slots and audit their frozen spatial readout."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import _base_training_data, identity, owner_lock
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg
from src.tokenizers.spatial_temporal_dictionary import SpatialTemporalDictionary, load_model
from src.tokenizers.supervised_waveform_dictionary import load_model as load_source_model

TRAIN_ROWS = None
TRAIN_SCALES = None


def write(path, value):
    def serial(item):
        if isinstance(item, dict):
            return {str(k): serial(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [serial(v) for v in item]
        if isinstance(item, np.generic):
            return serial(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def new_statistics(positions, slots):
    roles = np.arange(positions) % slots
    grams, crosses = {}, {}
    for mode in ('continuous', 'quantized'):
        for branch in ('S', 'O'):
            for slot in range(slots):
                count = int(np.sum(roles == slot))
                grams[mode, branch, slot] = np.zeros((positions, count, count))
                crosses[mode, branch, slot] = np.zeros((positions, count))
    return grams, crosses


def average_statistics(statistics, availability):
    denominator = np.sum(availability, axis=0)
    result = {}
    for key in statistics[0]:
        arrays = np.stack([item[key] for item in statistics])
        weights = np.asarray(availability).reshape(*np.asarray(availability).shape, *([1]*(arrays.ndim-2)))
        total = np.sum(arrays*weights, axis=0)
        shape = (len(denominator),)+((1,)*(total.ndim-1))
        result[key] = total/np.maximum(denominator, 1).reshape(shape)
    return result, denominator > 0


def fit_dataset(task):
    dataset, subjects, cfg, metadata, output, *model_path = task
    rows, scales = TRAIN_ROWS, TRAIN_SCALES
    if rows is None or scales is None:
        raise ValueError('forked read-only G0 metadata required')
    folder = output/dataset
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'weights.npz'
    if checkpoint.exists():
        stored = json.loads((folder/'fit_summary.json').read_text())
        if stored['config_identity'] != identity(cfg):
            raise ValueError('retained spatial-temporal fit configuration changed')
        return dataset
    started = time.monotonic()
    if model_path:
        from src.tokenizers.spatial_semantic_dictionary import load_model as load_semantic
        model = load_semantic(model_path[0])
    else:
        source = load_source_model(ROOT/cfg['model']['source_checkpoint'])
        model = SpatialTemporalDictionary(metadata, source, {})
    names = metadata['positions'][dataset]
    positions = len(names)
    cache = OrderedDict()
    subject_grams, subject_crosses, subject_support = [], [], []
    support = []
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        for subject, crops in sorted(subjects.items()):
            records = defaultdict(list)
            for crop in crops:
                records[rows[crop['row']]['record']].append(crop)
            record_grams, record_crosses, record_support = [], [], []
            for record, record_crops in sorted(records.items()):
                grams, crosses = new_statistics(positions, model.slots)
                counts = np.zeros(positions)
                for crop in record_crops:
                    fields, observed, target_valid = base_tokenizer_joint_eeg(rows, crop, scales, names, cache)
                    values = fields['x'].T.reshape(1, positions, 15, 400)
                    prediction = model.predict_continuous(values, dataset, observed[None])
                    ids = model.encode(values, dataset, observed[None])
                    sources = dict(continuous=model.slot_sources(prediction, observed[None], dataset),
                                   quantized=model.integer_sources(ids, dataset))
                    counts += target_valid
                    support.append(dict(subject=subject, record=record, **crop,
                        observed_positions=int(observed.sum()), complete_scored_positions=int(target_valid.sum())))
                    for mode in sources:
                        for branch in ('S', 'O'):
                            target = fields[branch.lower()].T.reshape(positions, 15, 400)
                            for slot in range(model.slots):
                                begin = slot*model.slot_samples
                                x = sources[mode][branch, slot][0].transpose(1, 2, 0).reshape(-1, int(np.sum(model.roles(dataset) == slot)))
                                y = target[:, :, begin:begin+model.slot_samples].transpose(1, 2, 0).reshape(-1, positions)
                                grams[mode, branch, slot][target_valid] += x.T@x/len(x)
                                crosses[mode, branch, slot][target_valid] += (x.T@y/len(x))[:, target_valid].T
                for key in grams:
                    grams[key] /= np.maximum(counts, 1)[:, None, None]
                    crosses[key] /= np.maximum(counts, 1)[:, None]
                record_grams.append(grams); record_crosses.append(crosses); record_support.append(counts > 0)
            grams, available = average_statistics(record_grams, record_support)
            crosses, _ = average_statistics(record_crosses, record_support)
            subject_grams.append(grams); subject_crosses.append(crosses); subject_support.append(available)
            write(folder/'progress.json', dict(stage='train_fit', completed_subjects=len(subject_grams),
                   planned_subjects=len(subjects), elapsed_seconds=time.monotonic()-started))
        grams, supported = average_statistics(subject_grams, subject_support)
        crosses, _ = average_statistics(subject_crosses, subject_support)
        arrays = {}
        for key, gram in grams.items():
            count = gram.shape[-1]
            strength = cfg['training']['spatial_ridge_fraction']*np.trace(gram, axis1=1, axis2=2)/count+1e-12
            diagonal = np.arange(count)
            gram[:, diagonal, diagonal] += strength[:, None]
            weights = np.linalg.solve(gram, crosses[key][..., None])[..., 0].T
            arrays['_'.join(map(str, key))] = weights
        temporary = checkpoint.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **arrays)
        temporary.replace(checkpoint)
    write(folder/'fit_summary.json', dict(status='completed', dataset=dataset, config_identity=identity(cfg),
        native_positions=names, roles=model.roles(dataset), training_subjects=len(subjects),
        training_crops=len(support), target_positions_supported=supported, support=support,
        elapsed_seconds=time.monotonic()-started))
    return dataset


def add_slot_boundaries(audited, prediction, batch, slot_samples):
    mask = batch['mask'].reshape(len(batch['x']), 6000, 1)
    for label, interior in (('slot_jump_difference_nmse', False), ('interior_slot_jump_difference_nmse', True)):
        result = []
        bounds = np.arange(slot_samples, 6000, slot_samples)
        if interior:
            bounds = bounds[bounds % 400 != 0]
        valid = mask[:, bounds] & mask[:, bounds-1]
        for output, target in (('sum', 'x'), ('S', 's'), ('O', 'o')):
            error = (prediction[output]-batch[target]).reshape(len(mask), 6000, 1)
            delta = error[:, bounds]-error[:, bounds-1]
            count = valid.sum(1)
            mse = np.divide(np.where(valid, delta**2, 0.).sum(1), count,
                            out=np.full(count.shape, np.nan), where=count > 0)
            result.append(mse/batch['dv'+target][:, 0, :1])
        audited['metrics'][label] = np.stack(result, axis=1)


def screen(model, cfg, rows, scales, output, *, model_loader=load_model):
    panel_path = ROOT/cfg['evaluation']['panel']
    panel_metadata = json.loads(panel_path.with_name('panel.json').read_text())
    if (panel_metadata['panel_sha256'] != hashlib.sha256(panel_path.read_bytes()).hexdigest()
            or panel_metadata['cfg_identity'] != model.metadata['cfg_identity']
            or panel_metadata['scales_identity'] != model.metadata['scales_identity']):
        raise ValueError('exact registered screening panel required before arrays')
    with np.load(panel_path) as archive:
        examples = archive['eeg_screen'].copy()
    grouped = defaultdict(list)
    for ordinal, (index, start, position) in enumerate(examples):
        row = rows[index]
        if (row['split'] != 'validation' or row['modality'] != 'eeg'
                or row['subject'] == panel_metadata['confirmation_subjects'][row['dataset']]):
            raise ValueError('screening-only independent subject boundary before arrays')
        grouped[int(index), int(start)].append((ordinal, int(position)))
    cache = OrderedDict()
    parts = {mode: (defaultdict(list), defaultdict(list)) for mode in cfg['evaluation']['modes']}
    integer_parts = {}; integer_metadata = []; order = []; figures = []
    checkpoint = output/'model.npz'
    restored = model_loader(checkpoint)
    semantic = model.metadata['schema'] == 'spatial_semantic_dictionary_v1'
    rate = []
    for crop_number, ((index, start), targets) in enumerate(grouped.items()):
        row = rows[index]; dataset = row['dataset']; names = model.metadata['positions'][dataset]
        item = next(item for item in row['inputs']['score'] if item['start_s'] == start)
        crop = dict(row=index, start_s=start, positions=item['positions'])
        fields, observed, _ = base_tokenizer_joint_eeg(rows, crop, scales, names, cache, allowed_splits=('validation',))
        values = fields['x'].T.reshape(1, len(names), 15, 400)
        source = model.predict_continuous(values, dataset, observed[None])
        ids = model.encode(values, dataset, observed[None])
        decode_options = dict(calibrated=False) if semantic else {}
        hard = model.decode_ids(ids, dataset, **decode_options)
        repeated = restored.decode_ids({key: value.copy() for key, value in ids.items()}, dataset, **decode_options)
        for branch in ('S', 'O', 'sum'):
            np.testing.assert_array_equal(hard[branch], repeated[branch])
        continuous = model.decode_sources(model.slot_sources(source, observed[None], dataset), dataset, 'continuous')
        predictions = dict(source_Wiener=source, continuous_slots=continuous, quantized_slots=repeated)
        if semantic:
            predictions['calibrated_slots'] = restored.decode_ids(ids, dataset, calibrated=True)
        selected_examples = [(index, start, position) for _, position in targets]
        batch = base_tokenizer_batch(rows, selected_examples, scales, array_cache=cache)
        lookup = {name.upper(): i for i, name in enumerate(names)}
        selected = [lookup[row['positions'][position].upper()] for _, position in targets]
        original_scales = np.asarray([scales['positions']['eeg'][dataset][row['positions'][position]]['scale'] for _, position in targets])
        order.extend(ordinal for ordinal, _ in targets)
        for mode, full_prediction in predictions.items():
            prediction = {key: value[0, selected] for key, value in full_prediction.items()}
            audited = joint_reconstruction_metrics(prediction, batch, 1, original_scales)
            add_slot_boundaries(audited, prediction, batch, model.slot_samples)
            for key, value in audited['metrics'].items():
                parts[mode][0][key].append(value)
            for key, value in audited['support'].items():
                parts[mode][1][key].append(value)
        for branch in ('S', 'O'):
            integer_parts[f'{branch}_{crop_number}'] = ids[branch]
        integer_metadata.append(dict(dataset=dataset, subject=row['subject'], record=row['record'], row=index,
                                     start_s=start, native_positions=names, selected_examples=selected_examples))
        count = sum(value.size for value in ids.values())
        nonmissing = sum(int(np.count_nonzero(value)) for value in ids.values())
        rate.append(dict(dataset=dataset, row=index, start_s=start, native_position_count=len(names),
                         observed_position_count=int(observed.sum()), integer_count_all_auxiliary_positions=count,
                         nonmissing_integer_count=nonmissing, dense_fixed_length_bits=count*11,
                         dense_fixed_length_bits_per_second=count*11/30,
                         requested_target_positions=len(selected)))
        if dataset not in {example['dataset'] for example in figures}:
            figures.append(dict(dataset=dataset, subject=row['subject'], position=row['positions'][targets[0][1]],
                source_unit=row['source_unit'], start_s=start, scale=original_scales[0],
                mask=batch['mask'][0].reshape(-1), interface_eligible=model.metadata['interface_eligible'],
                slot_seconds=model.slot_samples/200,
                target={key: batch[key][0].reshape(-1) for key in ('x', 's', 'o')},
                prediction={mode: {key: value[0, selected[0]].reshape(-1) for key, value in p.items()}
                            for mode, p in predictions.items()}))
        write(output/'screen_progress.json', dict(completed_crops=crop_number+1, planned_crops=len(grouped)))
    np.savez_compressed(output/'integer_ids.npz', **integer_parts)
    write(output/'integer_metadata.json', integer_metadata)
    with np.load(output/'integer_ids.npz') as stored:
        loaded_ids = {key: stored[key].copy() for key in stored.files}
    replay_modes = ('quantized_slots', 'calibrated_slots') if semantic else ('quantized_slots',)
    replay_parts = {mode: defaultdict(list) for mode in replay_modes}
    for crop_number, ((index, start), targets) in enumerate(grouped.items()):
        row = rows[index]; names = model.metadata['positions'][row['dataset']]
        batch = base_tokenizer_batch(rows, [(index, start, position) for _, position in targets], scales, array_cache=cache)
        selected = [next(i for i, name in enumerate(names) if name.upper() == row['positions'][p].upper()) for _, p in targets]
        original_scales = np.asarray([scales['positions']['eeg'][row['dataset']][row['positions'][p]]['scale'] for _, p in targets])
        for mode in replay_modes:
            options = dict(calibrated=mode == 'calibrated_slots') if semantic else {}
            replay = restored.decode_ids({branch: loaded_ids[f'{branch}_{crop_number}'] for branch in ('S', 'O')}, row['dataset'], **options)
            prediction = {key: value[0, selected] for key, value in replay.items()}
            audited = joint_reconstruction_metrics(prediction, batch, 1, original_scales)
            add_slot_boundaries(audited, prediction, batch, model.slot_samples)
            for key, value in audited['metrics'].items():
                replay_parts[mode][key].append(value)
    ordering = np.argsort(order)
    summaries = {}
    for mode, (metrics, support) in parts.items():
        metrics = {key: np.concatenate(value)[ordering] for key, value in metrics.items()}
        support = {key: np.concatenate(value)[ordering] for key, value in support.items()}
        if mode in replay_parts:
            for key, value in metrics.items():
                np.testing.assert_allclose(value, np.concatenate(replay_parts[mode][key])[ordering], rtol=0, atol=0, equal_nan=True)
        np.savez_compressed(output/(mode+'_metrics.npz'), examples=examples, **metrics,
                            **{'support_'+key: value for key, value in support.items()})
        summaries[mode] = joint_reconstruction_summary(metrics, support, examples.tolist(), rows)
    write(output/'rate.json', dict(emitted_K=1025, source_K=1024, missing_ID=0,
        bits_per_ID_capacity=np.log2(1025), bits_per_ID_fixed_length=11,
        bits_per_branch_position_second=5.5, all_auxiliary_position_costs_included=True,
        stored_integer_file_bytes=(output/'integer_ids.npz').stat().st_size, by_crop=rate))
    render_examples(figures, output)
    write(output/'screen_summary.json', dict(status='completed', summaries=summaries,
        planned_examples=len(examples), jointly_encoded_crops=len(grouped),
        integer_replay=dict(status='passed', saved_integer_file='integer_ids.npz',
                            independent_model_reload=True, maximum_metric_difference=0),
        interface_eligible=model.metadata['interface_eligible'],
        encoded_support='full 2s attributes and waveform atom' if semantic else 'one 0.5s complementary time slot'))
    return summaries


def render_examples(examples, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for example in examples:
        fig, axes = plt.subplots(3, 3, figsize=(15, 8), constrained_layout=True)
        t = np.arange(6000)/200
        for row, (branch, target_key) in enumerate((('S', 's'), ('O', 'o'), ('sum', 'x'))):
            target = example['target'][target_key]*example['scale']
            for column, interval in enumerate(((0, 30), (14, 16), (13.8, 14.8))):
                axis = axes[row, column]
                axis.plot(t, target, color='black', linewidth=.8, label='Target')
                curves = [('source_Wiener', '#3073af'), ('continuous_slots', '#e6992a'), ('quantized_slots', '#cf4c49')]
                if 'calibrated_slots' in example['prediction']:
                    curves = curves[:-1]+[('quantized_slots', '#969696'), ('calibrated_slots', '#cf4c49')]
                for mode, colour in curves:
                    axis.plot(t, example['prediction'][mode][branch]*example['scale'], color=colour,
                              linewidth=.75, alpha=.9, label=mode)
                for boundary in np.arange(example.get('slot_seconds', .5), 30, example.get('slot_seconds', .5)):
                    axis.axvline(boundary, color='0.8', linewidth=.5, linestyle=':' if boundary % 2 else '-')
                axis.set_xlim(*interval)
                invalid = ~example.get('mask', np.ones(6000, bool))
                transitions = np.diff(np.r_[False, invalid, False].astype(int))
                for first, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                    axis.axvspan(first/200, end/200, color='0.8', alpha=.4)
                axis.set_xlabel('Time (s)'); axis.set_ylabel(branch+' ('+example['source_unit']+')')
                if row == 0 and column == 0:
                    axis.legend(fontsize=7)
        label = f"Full 2s semantic atoms with complementary {example.get('slot_seconds', .5):g}s carriers" if example.get('interface_eligible') else 'Complementary 0.5s slot diagnostic'
        fig.suptitle(f"{label} | {example['dataset']} / {example['subject']} / {example['position']} / crop {example['start_s']}s")
        fig.savefig(output/(example['dataset']+'_waveform.png'), dpi=220)
        plt.close(fig)


def synthetic(output):
    from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary
    rng = np.random.default_rng(83)
    source_metadata = dict(schema='supervised_waveform_dictionary_v1', modality='eeg', codebook_size=4,
                           atom_dim=100, interface_eligible=False)
    source = SupervisedWaveformDictionary(source_metadata,
        {b: np.full((3001, 1, 1), .5, complex) for b in ('S', 'O')},
        {b: np.zeros(100) for b in ('S', 'O')}, {b: np.eye(100) for b in ('S', 'O')},
        {b: rng.normal(size=(4, 100)) for b in ('S', 'O')})
    metadata = dict(schema='spatial_temporal_dictionary_v1', positions={'fixture': ['A', 'B', 'C', 'D']},
                    slot_samples=100, interface_eligible=False)
    weights = {('fixture', mode, b, slot): np.ones((1, 4)) for mode in ('continuous', 'quantized')
               for b in ('S', 'O') for slot in range(4)}
    model = SpatialTemporalDictionary(metadata, source, weights)
    x = rng.normal(size=(2, 4, 15, 400)); observed = np.ones((2, 4), bool); observed[1, 2] = False
    ids = model.encode(x, 'fixture', observed)
    assert ids['S'].shape == (2, 4, 15)
    assert np.all(ids['S'][1, 2] == 0)
    decoded = model.decode_ids(ids, 'fixture')
    np.testing.assert_array_equal(decoded['S'][1, :, :, 200:300], 0)
    # Distinct known atoms in each subslot verify that each position emits its
    # own assigned time, with the other three subslots absent from its ID.
    known = np.zeros((1, 4, 15, 400))
    for slot in range(4):
        known[:, :, :, slot*100:(slot+1)*100] = 2*source.atoms['S'][slot]
    known_ids = model.encode(known, 'fixture', np.ones((1, 4), bool))
    for position in range(4):
        np.testing.assert_array_equal(known_ids['S'][:, position], position+1)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'model.npz'; model.save(path); restored = load_model(path)
        for branch in ('S', 'O', 'sum'):
            np.testing.assert_array_equal(decoded[branch], restored.decode_ids(ids, 'fixture')[branch])
    write(output/'synthetic.json', dict(status='passed', integer_shape=[2, 4, 15],
        missing_role_produces_zero_slot=True, roundtrip_exact=True,
        source_input='X', decoder_inputs='integers and fixed model/namespace only'))


def main():
    global TRAIN_ROWS, TRAIN_SCALES
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_temporal_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        synthetic(args.output)
        return
    cfg = yaml.safe_load(args.config.read_text())
    if (cfg['schema'] != 'dual_component_spatial_temporal_v1' or cfg['model']['interface_eligible']
            or cfg['training']['fitting_splits'] != ['train'] or cfg['model']['slot_s'] != .5
            or cfg['evaluation']['role'] != 'screen' or cfg['model']['context_s'] != 30
            or cfg['model']['token_s'] != 2 or cfg['model']['input'] != 'observed_X_only'):
        raise ValueError('versioned X-only complementary-slot screening diagnostic required')
    if (args.output/'summary.json').exists():
        raise FileExistsError('completed spatial-temporal evidence is retained')
    with owner_lock(args.output):
        started = time.monotonic()
        manifest, scales, _, _ = _base_training_data(cfg, ROOT)
        rows = manifest['rows']
        TRAIN_ROWS, TRAIN_SCALES = rows, scales
        source = load_source_model(ROOT/cfg['model']['source_checkpoint'])
        if (source.metadata['input_identity'] != identity(manifest) or source.atom_dim != 100
                or source.codebook_size != cfg['model']['source_codebook_size']):
            raise ValueError('exact matched frozen half-second source dictionary required')
        with np.load(ROOT/cfg['model']['namespace_checkpoint']) as archive:
            namespace = json.loads(str(archive['metadata']))
        if namespace['input_identity'] != manifest['cfg_identity']:
            raise ValueError('exact G0 namespace reference required')
        plan = json.loads((ROOT/cfg['training']['spatial_plan']).read_text())['plan']
        if any(rows[crop['row']]['split'] != 'train' or rows[crop['row']]['dataset'] != dataset
               for dataset, subjects in plan.items() for crops in subjects.values() for crop in crops):
            raise ValueError('training-only spatial plan required before arrays')
        metadata = dict(schema='spatial_temporal_dictionary_v1', positions=namespace['positions'], slot_samples=100,
            context_s=30, token_s=2, source_K=1024, emitted_K=1025, reserved_missing_ID=0,
            integer_count_per_branch_position_window=15, fixed_length_bits_per_ID=11,
            roles='fixed_native_electrode_index_modulo_4', cfg_identity=manifest['cfg_identity'],
            scales_identity=manifest['scales_identity'], config_identity=identity(cfg),
            source_checkpoint=str(ROOT/cfg['model']['source_checkpoint']),
            source_checkpoint_sha256=hashlib.sha256((ROOT/cfg['model']['source_checkpoint']).read_bytes()).hexdigest(),
            input='X_only', interface_eligible=False, encoded_support_per_ID_seconds=.5)
        write(args.output/'resolved_config.json', dict(config=cfg, metadata=metadata, training_plan=plan))
        shutil.copy2(__file__, args.output/'source_execution.py')
        tasks = [(dataset, subjects, cfg, metadata, args.output) for dataset, subjects in plan.items()]
        with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers'],
                                 mp_context=multiprocessing.get_context('fork')) as pool:
            futures = [pool.submit(fit_dataset, task) for task in tasks]
            for future in as_completed(futures):
                print(json.dumps(dict(dataset=future.result(), stage='fit_completed')), flush=True)
        weights = {}
        for dataset in plan:
            with np.load(args.output/dataset/'weights.npz') as archive:
                for mode in ('continuous', 'quantized'):
                    for branch in ('S', 'O'):
                        for slot in range(4):
                            weights[dataset, mode, branch, slot] = archive[f'{mode}_{branch}_{slot}']
        model = SpatialTemporalDictionary(metadata, source, weights)
        model.save(args.output/'model.npz')
        with threadpool_limits(cfg['resources']['screen_threads']):
            summaries = screen(model, cfg, rows, scales, args.output)
        write(args.output/'summary.json', dict(status='completed', metadata=metadata,
            elapsed_seconds=time.monotonic()-started,
            screen_scores={key: value['selection_score'] for key, value in summaries.items()},
            checkpoint='model.npz', screen_evidence='screen_summary.json',
            interface_eligible=False, decoder_continuous_side_information=0))


if __name__ == '__main__':
    main()
