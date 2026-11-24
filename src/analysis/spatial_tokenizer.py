"""Frozen multi-position EEG integer audit and waveform exports."""
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg
from src.tokenizers.spatial_waveform_dictionary import load_model

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, default=encode) + '\n')
    temporary.replace(path)


def edge_metrics(prediction, batch):
    mask = batch['mask'].reshape(len(batch['x']), 6000)
    groups = dict(head_2s_nmse=np.arange(400), tail_2s_nmse=np.arange(5600, 6000),
                  endpoints_2s_nmse=np.r_[np.arange(400), np.arange(5600, 6000)],
                  interior_26s_nmse=np.arange(400, 5600))
    result = {name: [] for name in groups}
    for branch, field in (('sum', 'x'), ('S', 's'), ('O', 'o')):
        truth = batch[field].reshape(mask.shape)
        estimate = prediction[branch].reshape(mask.shape)
        error = np.where(mask, estimate - truth, 0.)
        normalized = np.square(error) / batch['v' + field].reshape(mask.shape)
        for name, indices in groups.items():
            count = mask[:, indices].sum(axis=1)
            value = np.divide(normalized[:, indices].sum(axis=1), count,
                              out=np.full(len(mask), np.nan), where=count > 0)
            result[name].append(value)
    return {name: np.stack(values, axis=1)[..., None] for name, values in result.items()}


def screen(model, checkpoint, rows, scales, cfg, output, ids_source=None, *,
           decoder_options=None, intervention=None, mode_names=('independent', 'spatial'), render=True):
    decoder_options = decoder_options or {}
    if intervention not in (None, 'S_only', 'O_only', 'S_shift_14s', 'O_shift_14s', 'S_mask_alternating', 'O_mask_alternating'):
        raise ValueError('registered integer intervention required')
    panel_path = ROOT / cfg['evaluation']['panel']
    panel_metadata = json.loads(panel_path.with_name('panel.json').read_text())
    if panel_metadata['panel_sha256'] != hashlib.sha256(panel_path.read_bytes()).hexdigest():
        raise ValueError('registered panel changed')
    if (panel_metadata['cfg_identity'] != model.metadata['input_identity']
            or panel_metadata['scales_identity'] != model.metadata['scales_identity']):
        raise ValueError('screen panel G0 identity mismatch before arrays')
    with np.load(panel_path) as stored:
        examples = stored['eeg_' + cfg['evaluation']['role']].copy()
    grouped = defaultdict(list)
    for ordinal, (index, start, position) in enumerate(examples):
        if rows[index]['split'] != 'validation':
            raise ValueError('registered validation screen required')
        grouped[int(index), int(start)].append((ordinal, int(position)))
    retained_ids = None
    retained_metadata = None
    if ids_source is not None:
        with np.load(ids_source, allow_pickle=False) as stored:
            retained_ids = {key: stored[key].copy() for key in stored.files}
        retained_metadata = json.loads(ids_source.with_name('integer_metadata.json').read_text())
        if len(retained_ids) != 2 * len(grouped) or len(retained_metadata) != len(grouped):
            raise ValueError('saved integer panel support changed')
    cache = OrderedDict()
    modes = {mode: (defaultdict(list), defaultdict(list)) for mode in mode_names}
    saved_ids = {}
    id_metadata = []
    ordinal_order = []
    replay_model = load_model(checkpoint)
    replay_maximum = 0.
    figures = []
    for crop_number, ((index, start), targets) in enumerate(grouped.items()):
        row = rows[index]
        dataset = row['dataset']
        names = model.metadata['positions'][dataset]
        if retained_ids is None:
            item = next(item for item in row['inputs']['score'] if item['start_s'] == start)
            crop = dict(row=index, start_s=start, positions=item['positions'])
            fields, observed, _ = base_tokenizer_joint_eeg(rows, crop, scales, names, cache, allowed_splits=('validation',))
            values = fields['x'].T.reshape(1, len(names), 15, 400)
            ids = model.encode(values, dataset, observed[None])
        else:
            retained = retained_metadata[crop_number]
            if (retained['row'] != index or retained['start_s'] != start or retained['dataset'] != dataset
                    or retained['native_positions'] != names):
                raise ValueError('saved integer native crop identity changed')
            ids = {branch: retained_ids[f'{branch}_{crop_number}'] for branch in ('S', 'O')}
        if intervention is not None:
            ids = {branch: value.copy() for branch, value in ids.items()}
            if intervention in ('S_only', 'O_only'):
                ids['O' if intervention == 'S_only' else 'S'].fill(0)
            else:
                branch = intervention[0]
                if intervention.endswith('shift_14s'):
                    ids[branch] = np.roll(ids[branch], 7, axis=-1)
                else:
                    ids[branch][..., ::2] = 0
        id_metadata.append(dict(crop=crop_number, dataset=dataset, subject=row['subject'], record=row['record'],
                                row=index, start_s=start, native_positions=names))
        for branch in ('S', 'O'):
            saved_ids[f'{branch}_{crop_number}'] = ids[branch]
        selected_examples = [(index, start, position) for _, position in targets]
        batch = base_tokenizer_batch(rows, selected_examples, scales, array_cache=cache)
        selected_indices = [next(j for j, name in enumerate(names) if name.upper() == row['positions'][p].upper()) for _, p in targets]
        original_scales = np.array([scales['positions']['eeg'][dataset][row['positions'][p]]['scale'] for _, p in targets])
        ordinal_order.extend(ordinal for ordinal, _ in targets)
        for mode in modes:
            prediction = model.decode_ids(ids, dataset, spatial=mode == 'spatial', **decoder_options)
            repeated = replay_model.decode_ids({branch: saved_ids[f'{branch}_{crop_number}'] for branch in ('S', 'O')},
                                                dataset, spatial=mode == 'spatial', **decoder_options)
            for branch in ('S', 'O', 'sum'):
                np.testing.assert_array_equal(prediction[branch], repeated[branch])
                replay_maximum = max(replay_maximum, float(np.max(abs(prediction[branch] - repeated[branch]))))
            selected = {branch: repeated[branch][0, selected_indices] for branch in ('S', 'O', 'sum')}
            audited = joint_reconstruction_metrics(selected, batch, 1, original_scales)
            audited['metrics'].update(edge_metrics(selected, batch))
            for key, value in audited['metrics'].items():
                modes[mode][0][key].append(value)
            for key, value in audited['support'].items():
                modes[mode][1][key].append(value)
            if dataset not in {item['dataset'] for item in figures} and mode == 'spatial':
                independent = model.decode_ids(ids, dataset, spatial=False)
                figures.append(dict(dataset=dataset, subject=row['subject'], record=row['record'], start_s=start,
                    position=row['positions'][targets[0][1]], scale=float(original_scales[0]), source_unit=row['source_unit'],
                    scoring_mask=batch['mask'][0].reshape(-1).copy(),
                    target={branch: batch[branch.lower()][0].reshape(-1).copy() for branch in ('S', 'O')},
                    independent={branch: independent[branch][0, selected_indices[0]].reshape(-1).copy() for branch in ('S', 'O')},
                    spatial={branch: selected[branch][0].reshape(-1).copy() for branch in ('S', 'O')}))
        write(output / 'evaluation_progress.json', dict(completed_crops=crop_number + 1, planned_crops=len(grouped)))
    np.savez_compressed(output / 'integer_ids.npz', **saved_ids)
    with np.load(output / 'integer_ids.npz') as stored:
        for key, value in saved_ids.items():
            np.testing.assert_array_equal(stored[key], value)
        replay_codes = {key: stored[key].copy() for key in stored.files}
    write(output / 'integer_metadata.json', id_metadata)
    replay_model = load_model(checkpoint)
    replay_metrics = {mode: defaultdict(list) for mode in modes}
    for crop_number, ((index, start), targets) in enumerate(grouped.items()):
        row = rows[index]
        dataset = row['dataset']
        names = model.metadata['positions'][dataset]
        batch = base_tokenizer_batch(rows, [(index, start, position) for _, position in targets], scales,
                                     array_cache=cache, cache_limit=8)
        selected_indices = [next(j for j, name in enumerate(names) if name.upper() == row['positions'][p].upper())
                            for _, p in targets]
        original_scales = np.array([scales['positions']['eeg'][dataset][row['positions'][p]]['scale'] for _, p in targets])
        codes = {branch: replay_codes[f'{branch}_{crop_number}'] for branch in ('S', 'O')}
        for mode in modes:
            prediction = replay_model.decode_ids(codes, dataset, spatial=mode == 'spatial', **decoder_options)
            selected = {branch: prediction[branch][0, selected_indices] for branch in ('S', 'O', 'sum')}
            audited = joint_reconstruction_metrics(selected, batch, 1, original_scales)
            audited['metrics'].update(edge_metrics(selected, batch))
            for key, value in audited['metrics'].items():
                replay_metrics[mode][key].append(value)
    ordering = np.argsort(ordinal_order)
    summaries = {}
    for mode, (metrics, support) in modes.items():
        metrics = {key: np.concatenate(value)[ordering] for key, value in metrics.items()}
        for key, value in metrics.items():
            replay_value = np.concatenate(replay_metrics[mode][key])[ordering]
            np.testing.assert_allclose(value, replay_value, rtol=0, atol=0, equal_nan=True)
            metrics[key] = replay_value
        support = {key: np.concatenate(value)[ordering] for key, value in support.items()}
        np.savez_compressed(output / (mode + '_metrics.npz'), examples=examples, **metrics,
                            **{'support_' + key: value for key, value in support.items()})
        summaries[mode] = joint_reconstruction_summary(metrics, support, examples.tolist(), rows)
    write(output / 'screen_summary.json', dict(status='completed', summaries=summaries,
        planned_examples=len(examples), jointly_encoded_crops=len(grouped), integer_replay_max_abs=replay_maximum,
        same_integer_sequences=True, standalone_metric_replay='exact_from_saved_ID_file_after_model_reload',
        checkpoint=str(checkpoint), role=cfg['evaluation']['role'], intervention=intervention,
        decoder_options=decoder_options))
    if render:
        render_examples(figures, output)
    return summaries


def completion(model, checkpoint, summaries, started, audit_output):
    return dict(status='completed', checkpoint=str(checkpoint), metadata=model.metadata,
        screen_scores={mode: summary['selection_score'] for mode, summary in summaries.items()},
        elapsed_seconds=time.monotonic() - started, audit_output=str(audit_output),
        declaration='EEG-only candidate for joint baseline comparison',
        frozen_parameter_count=int(sum(value.size for value in model.atoms.values()) + sum(value.size for value in model.centers.values())
                                   + sum(value.size for value in model.weights.values()) + sum(value.size for value in model.transfer.values())))


def render_examples(examples, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for example in examples:
        for figure_name, intervals in [('waveform', ((0, 30), (10, 12), (11.7, 12.3))),
                                       ('endpoint_waveform', ((0, 2), (28, 30), (11.7, 12.3)))]:
            fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
            for row, branch in enumerate(('S', 'O')):
                target = example['target'][branch] * example['scale']
                for column, interval in enumerate(intervals):
                    ax = axes[row, column]
                    time_axis = np.arange(6000) / 200
                    for label, values in [('target', target), ('independent', example['independent'][branch] * example['scale']),
                                          ('spatial', example['spatial'][branch] * example['scale'])]:
                        ax.plot(time_axis, values, label=label, linewidth=.8, alpha=.85)
                    ax.set_xlim(*interval)
                    visible = (time_axis >= interval[0]) & (time_axis < interval[1])
                    arrays = [target, example['independent'][branch] * example['scale'], example['spatial'][branch] * example['scale']]
                    minimum = min(float(value[visible].min()) for value in arrays)
                    maximum = max(float(value[visible].max()) for value in arrays)
                    ax.set_ylim(minimum - .05 * (maximum - minimum + 1e-9), maximum + .05 * (maximum - minimum + 1e-9))
                    for boundary in np.arange(2, 30, 2):
                        ax.axvline(boundary, color='grey', linewidth=.5, linestyle=':')
                    invalid = ~example['scoring_mask']
                    transitions = np.diff(np.r_[False, invalid, False].astype(int))
                    for begin, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                        ax.axvspan(begin / 200, end / 200, color='grey', alpha=.25)
                    ax.set_xlabel('Window time (s)')
                    ax.set_ylabel(branch + ' (' + example['source_unit'] + ')')
                    if column == 0:
                        ax.legend(fontsize=8)
            fig.suptitle(f"{example['dataset']} / {example['subject']} / {example['record']} / {example['position']} / crop {example['start_s']} s\nDotted lines mark token boundaries; grey shading marks unscored samples")
            fig.savefig(output / (example['dataset'] + '_' + figure_name + '.png'), dpi=200)
            plt.close(fig)
    write(output / 'waveform_examples.json', [{key: value for key, value in example.items()
        if key not in ('target', 'independent', 'spatial', 'scoring_mask')} for example in examples])
