"""Independently replay the fixed training panel of montage word search and render voltage."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.band_exchange_words import load_fold_books
from src.tokenizers.cross_branch_innovations import posterior_waveforms
from src.utils.io import save_npz_atomic

STATE = {}
ARMS = ('frozen', 'semantic64', 'waveform')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def write(path, value):
    def scalar(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack14(values):
    a = np.asarray(values)
    if a.dtype.kind not in 'ui' or np.any(a < 0) or np.any(a >= 15351):
        raise ValueError('K15351 real integers required')
    output = bytearray(); buffer = 0; bits = 0
    for value in a.reshape(-1):
        buffer |= int(value) << bits; bits += 14
        while bits >= 8:
            output.append(buffer & 255); buffer >>= 8; bits -= 8
    if bits: output.append(buffer)
    return bytes(output)


def unpack14(payload, shape):
    count = int(np.prod(shape))
    if len(payload) != (count * 14 + 7) // 8: raise ValueError('payload length')
    output = np.empty(count, np.uint16); cursor = 0; buffer = 0; bits = 0
    for j in range(count):
        while bits < 14:
            buffer |= payload[cursor] << bits; cursor += 1; bits += 8
        output[j] = buffer & 16383; buffer >>= 14; bits -= 14
    if buffer or np.any(output >= 15351): raise ValueError('padding or unused integer')
    return output.reshape(shape)


def integer_atoms(sent, pilot_books, residual_books, state):
    """Fresh ID lookup, without encoder, source estimate, input signal or producer lookup."""
    n = len(sent['S'])
    pilots = np.concatenate((np.arange(n) % 2 == 0, np.arange(n) % 2 == 1))
    first = np.zeros((2 * n, 15, 400)); second = np.zeros_like(first)
    role = np.full(2 * n, -1, np.int32)
    for bi, branch in enumerate('SO'):
        values = np.asarray(sent[branch])
        if values.shape != (n, 15) or values.dtype.kind not in 'ui' or np.any(values >= 15351) or np.any(values < 0):
            raise ValueError('one valid integer per native site and slot')
        for site, stream in enumerate(values):
            index = bi * n + site
            if not np.any(stream): continue
            if np.any(stream == 0): raise ValueError('partially missing native site')
            if pilots[index]:
                if np.any(stream >= 5116): raise ValueError('residual word at paid pilot')
                tags = (stream.astype(np.int32) - 1) // 1023
                ids = (stream.astype(np.int32) - 1) % 1023
                band = int(tags[0]); book = pilot_books[branch][band]
                first[index] = book['balanced_atoms'][ids].astype(float) * book['balanced_gain'][ids, None]
            else:
                if np.any(stream < 5116): raise ValueError('pilot word at paid residual site')
                tags = (stream.astype(np.int32) - 5116) // 2047
                ids = (stream.astype(np.int32) - 5116) % 2047
                band = int(tags[0]); book = residual_books[branch][band]
                active = ids < len(book['innovation_atoms'])
                wave = np.zeros((15, 400)); wave[active] = book['innovation_atoms'][ids[active]]
                second[index] = wave * state['scale'][band, index]
            if not np.all(tags == band): raise ValueError('paid frequency role changed inside context')
            role[index] = band
    if not np.array_equal(role[:n] < 0, role[n:] < 0): raise ValueError('paired missing sites differ')
    return first, second, role


def decode_integers(sent, pilot_books, residual_books, state):
    first, second, role = integer_atoms(sent, pilot_books, residual_books, state)
    return posterior_waveforms(first, second, role, state['joint'], state['pilot_gain'], state['pilot_noise'],
                              state['innovation_mean_gain'], state['innovation_mean_noise'], 'innovation')


def budget():
    unit = os.environ.get('MONTAGE_REVIEW_UNIT')
    if unit:
        used = int(subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'CPUUsageNSec', '--value'], text=True)) / 1e9
        if used >= 1170: raise RuntimeError('20 CPU minute independent audit budget exhausted')


def metadata(identity, reason, number):
    row, start, position = map(int, identity); r = STATE['rows'][row]
    return dict(row=row, start_s=start, position=position, dataset=r['dataset'], subject=r['subject'], record=r['record'],
                parent=r['parent'], position_name=r['positions'][position], source_unit=r['source_unit'],
                scale=STATE['scales']['positions']['eeg'][r['dataset']][r['positions'][position]]['scale'],
                audit_selection=list(reason), audit_index=number)


def select():
    draws = json.loads((STATE['run'] / 'component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json').read_text())['examples']
    if len(draws) != 17: raise ValueError('original seventeen display identities required')
    chosen = {}
    for c in draws:
        key = tuple(c[k] for k in ('row', 'start_s', 'position'))
        chosen[key] = metadata(key, ['original17:' + x for x in c['selection']], len(chosen))
    for dataset in STATE['cases']:
        fields = defaultdict(list)
        for fold in range(4):
            with np.load(STATE['model'] / dataset / f'held_fold{fold}_scores.npz') as z:
                for k in ('examples', 'semantic64__nmse', 'waveform__nmse'): fields[k].append(z[k])
        examples = np.concatenate(fields['examples'])
        for arm in ARMS[1:]:
            score = np.concatenate(fields[arm + '__nmse'])
            for label, values in [('joint', score.mean((1, 2))), ('S', score[:, 1, 0]), ('O', score[:, 2, 0])]:
                key = tuple(map(int, examples[np.nanargmax(values)])); reason = f'{arm}:{label}_argmax_all_dataset_positions'
                if key in chosen: chosen[key]['audit_selection'].append(reason)
                else: chosen[key] = metadata(key, [reason], len(chosen))
    record = dict(examples=list(chosen.values()), original_count=17, selected_count=len(chosen),
                  rule=STATE['config']['selection'], source_score_files='held_fold{f}_scores.npz', scope='train_held_people_only')
    write(STATE['out'] / 'selected_examples.json', record)
    return chosen


def replay(dataset):
    rows, scales = STATE['rows'], STATE['scales']; cases = STATE['cases'][dataset]
    dest = STATE['out'] / dataset; dest.mkdir(exist_ok=True)
    if (dest / 'replay.json').exists(): return json.loads((dest / 'replay.json').read_text())
    before = time.process_time(); retained = {}; audit = []; allmetrics = defaultdict(list); allexamples = []
    selected = STATE['selected']; cache = OrderedDict()
    for fold in range(4):
        old = {b: load_fold_books(STATE['pilot'], fold, b) for b in 'SO'}
        new = {b: load_fold_books(STATE['frozen'] / 'books', fold, b) for b in 'SO'}
        with np.load(STATE['frozen'] / dataset / f'fold_{fold}_state.npz') as z: state = dict(z)
        fit = {c['subject'] for c in cases if c['fold'] != fold}; held = {c['subject'] for c in cases if c['fold'] == fold}
        if set(state['fit_people']) != fit or set(state['held_people']) != held or fit & held: raise ValueError('frozen fit/held identity changed')
        idfile = STATE['model'] / dataset / f'held_fold{fold}_ids.npz'
        with np.load(idfile) as z: sent = dict(z)
        with np.load(STATE['model'] / dataset / f'held_fold{fold}_scores.npz') as z: expected = dict(z)
        with np.load(STATE['frozen'] / dataset / f'fold_{fold}_ids.npz') as z: original = dict(z)
        original_index = {tuple(e): i for i, e in enumerate(original['examples'])}
        offset = 0; metrics = defaultdict(list); count = 0; words = 0; bytes_count = 0
        for c in cases:
            if c['fold'] != fold: continue
            budget(); take = np.arange(offset, offset + c['count']); offset += len(take)
            examples = np.array([(c['crop']['row'], c['crop']['start_s'], p) for p in c['crop']['positions']])
            np.testing.assert_array_equal(sent['examples'][take], examples)
            np.testing.assert_array_equal(expected['examples'][take], examples)
            native = np.asarray(c['native']); order = np.argsort(native)
            np.testing.assert_array_equal(native[order], np.arange(len(native)))
            original_take = np.array([original_index[tuple(e)] for e in examples])[order]
            streams = {'frozen': {b: original['innovation_' + b][original_take] for b in 'SO'}}
            streams.update({a: {b: sent[a + '_' + b][take[order]] for b in 'SO'} for a in ARMS[1:]})
            # Decode has no access to continuous source, target, training fits, or high-level encoder.
            predictions = {}
            for arm, stream in streams.items():
                values = np.stack([stream[b] for b in 'SO']); packed = pack14(values); restored = unpack14(packed, values.shape)
                np.testing.assert_array_equal(restored, values); words += values.size; bytes_count += len(packed)
                decoded = decode_integers(dict(zip('SO', restored)), old, new, state)
                predictions[arm] = {b: v[native] for b, v in decoded.items()}
                # Pure decoder contract: after immutable model/ID inputs are materialized, file reads are forbidden.
                if count == 0:
                    import builtins
                    original_open, original_load = builtins.open, np.load
                    def forbidden(*args, **kwargs): raise RuntimeError('decoder attempted an external read')
                    try:
                        builtins.open = forbidden; np.load = forbidden
                        checked = decode_integers(dict(zip('SO', restored)), old, new, state)
                    finally: builtins.open, np.load = original_open, original_load
                    for b in decoded: np.testing.assert_array_equal(checked[b], decoded[b])
            batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=4)
            factors = np.array([scales['positions']['eeg'][dataset][rows[i]['positions'][p]]['scale'] for i, _, p in examples])
            for arm, predicted in predictions.items():
                scored = joint_reconstruction_metrics(predicted, batch, 1, factors)
                if scored['support']['nonfinite_prediction_points'].any(): raise ValueError('nonfinite integer wave')
                support = batch['mask'].reshape(len(examples), 6000).astype(bool)
                for component, (output, target) in enumerate([('sum', 'x'), ('S', 's'), ('O', 'o')]):
                    y = batch[target].reshape(len(examples), 6000).astype(float)
                    p = predicted[output].reshape(len(examples), 6000)
                    variance = batch['v' + target][:, 0, 0]
                    if not np.all(np.isfinite(variance) & (variance > 0)): raise ValueError('invalid frozen target variance')
                    mse = np.where(support, (p - y) ** 2, 0).sum(1) / support.sum(1).clip(1)
                    np.testing.assert_allclose(mse / variance, scored['metrics']['nmse'][:, component, 0], rtol=1e-13, atol=1e-13)
                    np.testing.assert_allclose(np.sqrt(mse) * factors, scored['metrics']['rmse_native'][:, component, 0], rtol=1e-13, atol=1e-13)
                for name, value in scored['metrics'].items():
                    np.testing.assert_array_equal(value, expected[arm + '__' + name][take])
                    metrics[arm + '__' + name].append(value)
            for j, e in enumerate(examples):
                key = tuple(e)
                if key not in selected: continue
                index = selected[key]['audit_index']
                for k in ('x', 's', 'o', 'mask', 'vx', 'vs', 'vo'): retained[f'{index}_{k}'] = batch[k][j]
                for arm, predicted in predictions.items():
                    for k, value in predicted.items(): retained[f'{index}_{arm}_{k}'] = value[j]
                save_npz_atomic(dest / f'case_{index:03d}_full_native_ids.npz', **{a + '_' + b: v[b] for a, v in streams.items() for b in 'SO'})
            count += 1
        full = {k: np.concatenate(v) for k, v in metrics.items()}
        save_npz_atomic(dest / f'fold{fold}_replayed_metrics.npz', examples=sent['examples'], **full)
        allexamples.append(sent['examples'])
        for k, v in full.items(): allmetrics[k].append(v)
        audit.append(dict(fold=fold, crops=count, positions=offset, fit_people=len(fit), held_people=len(held),
                          integer_count_all3arms=words, packed_bytes_all3arms=bytes_count, saved_id_sha256=sha(idfile),
                          all_metric_arrays_bitwise_exact=True, decoder_no_file_reads_probe_all3arms=True))
    save_npz_atomic(dest / 'display_arrays.npz', **retained)
    combined = {k: np.concatenate(v) for k, v in allmetrics.items()}; examples = np.concatenate(allexamples)
    summaries = {a: base_training_aggregate({k.split('__', 1)[1]: v for k, v in combined.items() if k.startswith(a + '__')}, examples, rows) for a in ARMS}
    record = dict(dataset=dataset, status='passed', folds=audit, positions=len(examples), crops=sum(v['crops'] for v in audit),
                  CPU_seconds=time.process_time() - before, summaries=summaries)
    write(dest / 'replay.json', record); print(dataset, 'replay complete', record['CPU_seconds'], flush=True)
    return record


def render(dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    before = time.process_time(); dest = STATE['out']; folder = dest / STATE.get('figure_directory', 'figures'); folder.mkdir(exist_ok=True)
    with np.load(dest / dataset / 'display_arrays.npz') as z: arrays = dict(z)
    chosen = [c for c in STATE['selected'].values() if c['dataset'] == dataset]; records = []
    if STATE.get('X_events', False):
        chosen = [c for c in chosen if any(x in ('original17:reference_max', 'original17:candidate_max') for x in c['audit_selection'])]
    palette = {'target': '#202833', 'frozen': '#b98139', 'semantic64': '#2468b4', 'waveform': '#148269'}
    styles = {'target': '-', 'frozen': '--', 'semantic64': '-.', 'waveform': ':'}
    labels = {'target': 'measured target', 'frozen': 'v8 frozen actual IDs', 'semantic64': 'joint semantic64 IDs', 'waveform': 'joint waveform IDs'}
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    t = np.arange(6000) / 200
    for case in chosen:
        budget(); index = case['audit_index']; scale = float(case['scale'])
        waves = {'target': {b: arrays[f'{index}_{k}'].reshape(6000).astype(float) * scale for b, k in [('X', 'x'), ('S', 's'), ('O', 'o')]}}
        waves.update({a: {b: arrays[f'{index}_{a}_{k}'].reshape(6000) * scale for b, k in [('X', 'sum'), ('S', 'S'), ('O', 'O')]} for a in ARMS})
        valid = arrays[f'{index}_mask'].reshape(6000).astype(bool)
        event_branch = 'X' if STATE.get('X_events', False) else 'S'
        peak = int(np.argmax(np.where(valid, abs(waves['target'][event_branch]), -np.inf)))
        center = float(np.clip(peak / 200, 1, 29)); boundary = float(np.clip(2 * round(center / 2), 2, 28))
        unit = 'REFED export unit' if dataset == 'refed' else str(case['source_unit'])
        identity = f"{dataset} | {case['subject']} | {case['record']} | {case['position_name']} | crop +{case['start_s']}s"
        caption = 'Shared source coordinates; gray=original mask; lines=2s boundaries; no per-window rescaling'
        if dataset == 'refed': caption += '; REFED physical unit unknown'
        stem = f"{index:03d}_{dataset}_{case['subject']}_{case['position_name']}"
        voltage = folder / (stem + '_voltage.png'); dynamic = folder / (stem + '_dynamics.png')
        fig, axes = plt.subplots(3, 4, figsize=(17.5, 8.8), layout='constrained')
        windows = [(0, 30), (14, 16), (center - 1, center + 1), (boundary - .5, boundary + .5)]
        for row, branch in enumerate(('X', 'S', 'O')):
            extrema = np.array([v[branch] for v in waves.values()]); low, high = float(extrema.min()), float(extrema.max()); margin = .06 * (high - low or 1)
            for col, limits in enumerate(windows):
                ax = axes[row, col]
                for arm, values in waves.items(): ax.plot(t, values[branch], color=palette[arm], ls=styles[arm], lw=.85, label=labels[arm])
                for edge in range(0, 31, 2): ax.axvline(edge, color='#b7bec3', lw=.45, alpha=.65)
                ax.fill_between(t, low - margin, high + margin, where=~valid, color='#bfc3c7', alpha=.45)
                ax.set(xlim=limits, ylim=(low - margin, high + margin), xlabel='Time within 30s (s)')
                if col == 0: ax.set_ylabel(branch + ' / ' + unit)
                if row == 0: ax.set_title(['Full 200-Hz voltage', 'Fixed 14–16s voltage', '2s at true ' + event_branch + ' peak', 'Nearest 2s boundary'][col])
        axes[0, 0].legend(fontsize=6.5); fig.suptitle(identity + '\n' + caption, fontsize=10)
        fig.savefig(voltage, dpi=200, facecolor='white'); plt.close(fig)
        if STATE.get('X_events', False):
            records.append(dict(identity=case, voltage=str(voltage.relative_to(dest)), event_reference_branch='X',
                                X_event_peak_s=peak / 200, event_window_s=windows[2], boundary_s=boundary,
                                mask_valid_points=int(valid.sum()), status='rendered_pending_actual_open'))
            continue
        full = valid.reshape(15, 400).all(1); block = valid.reshape(120, 50).all(1)
        hz = np.fft.rfftfreq(400, 1 / 200); hann = np.hanning(400)
        fig, axes = plt.subplots(3, 4, figsize=(17.5, 8.8), layout='constrained'); metrics = {}
        for row, branch in enumerate(('X', 'S', 'O')):
            target = waves['target'][branch]; at = int(np.argmax(np.where(valid, abs(target), -np.inf)))
            near = np.arange(max(0, at - 100), min(6000, at + 101)); near = near[valid[near]]
            metrics[branch] = {'target_peak_time_s': at / 200, 'target_peak': float(target[at]), 'models': {}}
            for mi, (arm, values) in enumerate(waves.items()):
                y = values[branch]; env = np.sqrt((y.reshape(120, 50) ** 2).mean(1)); env[~block] = np.nan
                patches = y.reshape(15, 400)
                psd = abs(np.fft.rfft((patches - patches.mean(1, keepdims=True)) * hann, axis=-1)) ** 2 * 2 / (400 * (hann * hann).sum())
                energy = np.stack([psd[:, (hz >= lo) & ((hz <= hi) if hi == 45 else (hz < hi))].sum(1) for lo, hi in BANDS], 1); energy[~full] = np.nan
                axes[row, 0].plot(np.arange(120) * .25 + .125, env, color=palette[arm], ls=styles[arm], label=labels[arm])
                for col, band in [(1, 1), (2, 2)]: axes[row, col].plot(np.arange(15) * 2 + 1, energy[:, band], color=palette[arm], ls=styles[arm], marker='.', ms=3)
                avg = np.nanmean(energy, axis=0); axes[row, 3].bar(np.arange(5) + (mi - 1.5) * .19, np.maximum(avg, 1e-25), .18, color=palette[arm])
                k = int(near[np.argmax(abs(y[near]))]); mask = valid
                metrics[branch]['models'][arm] = dict(mean_band_energy=avg, complete_patches=int(full.sum()),
                    RMSE_native=float(np.sqrt(np.mean((y[mask] - target[mask]) ** 2))),
                    RMS_gain=float(np.sqrt(np.mean(y[mask] ** 2) / np.mean(target[mask] ** 2))),
                    target_peak_signed_ratio=float(y[at] / target[at]) if target[at] else None,
                    value_at_target_peak=float(y[at]), local_abs_peak=float(y[k]), local_peak_time_s=k / 200,
                    local_peak_delay_s=(k - at) / 200,
                    event2s_RMSE_native=float(np.sqrt(np.mean((y[int(center*200)-200:int(center*200)+200] - target[int(center*200)-200:int(center*200)+200]) ** 2))))
            for col in range(3): axes[row, col].set(xlim=(0, 30), xlabel='Time within 30s (s)', ylabel=branch + (' / ' + unit if col == 0 else ' / source unit²'))
            axes[row, 3].set(xticks=range(5), xticklabels=['1–4', '4–8', '8–13', '13–30', '30–45'], yscale='log', xlabel='Frequency (Hz)', ylabel=branch + ' / source unit²')
            if row == 0:
                for col, title in enumerate(['0.25s local RMS', '2s theta 4–8Hz power', '2s alpha 8–13Hz power', 'Mean complete-patch energy']): axes[row, col].set_title(title)
        axes[0, 0].legend(fontsize=6.5); fig.suptitle(identity + '\nOriginal complete-patch/block masks; source-unit² bands within 1–45Hz; unsmoothed', fontsize=10)
        fig.savefig(dynamic, dpi=200, facecolor='white'); plt.close(fig)
        records.append(dict(identity=case, voltage=str(voltage.relative_to(dest)), dynamics=str(dynamic.relative_to(dest)),
                            mask_valid_points=int(valid.sum()), source_unit_display=unit, S_event_peak_s=peak / 200,
                            event_window_s=windows[2], boundary_s=boundary, metrics=metrics, status='rendered_pending_actual_open'))
        write(dest / dataset / 'render_progress.json', dict(completed=len(records), planned=len(chosen)))
    write(dest / dataset / ('X_event_figure_manifest.json' if STATE.get('X_events', False) else 'figure_manifest.json'), records)
    print(dataset, 'rendered', len(records), 'CPU', time.process_time() - before, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'X-events'], default='all')
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    if cfg['data'] != {'fit': 'none_frozen_models', 'score': 'original_train_outer_held_people', 'validation': 'forbidden', 'confirmation': 'forbidden', 'test': 'forbidden'}: raise ValueError('train-only review contract')
    run = ROOT / cfg['run']; model = run / cfg['model']; out = model / 'visual_review_v1'; out.mkdir(exist_ok=True)
    pop = json.loads((run / cfg['population']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(yaml.safe_load((ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()), ROOT)
    if (pop['cfg_identity'], pop['scales_identity']) != (manifest['cfg_identity'], manifest['scales_identity']): raise ValueError('frozen G0 data or scales changed')
    if len(pop['cases']) != 400 or sum(c['count'] for c in pop['cases']) != 17040: raise ValueError('original population changed')
    rows = manifest['rows']
    for c in pop['cases']:
        r = rows[c['crop']['row']]
        if (r['split'], r['modality'], r['dataset'], r['subject']) != ('train', 'eeg', c['dataset'], c['subject']): raise ValueError('training boundary before native arrays')
    cases = {d: [c for c in pop['cases'] if c['dataset'] == d] for d in sorted(pop['native_positions'])}
    STATE.update(run=run, model=model, out=out, rows=rows, scales=scales, config=cfg, cases=cases,
                 pilot=run / cfg['pilot'], frozen=run / cfg['frozen'])
    if args.stage == 'X-events':
        selection = json.loads((out / 'selected_examples.json').read_text())['examples']
        STATE.update(selected={tuple(c[k] for k in ('row', 'start_s', 'position')): c for c in selection}, X_events=True, figure_directory='X_event_figures')
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool: records = list(pool.map(render, cases))
        records = sorted([r for group in records for r in group], key=lambda r: r['identity']['audit_index'])
        write(out / 'X_event_figure_manifest.json', dict(status='rendered_pending_actual_open', examples=records, figure_count=len(records),
              selection='same original17 reference_max/candidate_max identities; no additional member selection'))
        return
    write(out / 'resolved_audit_config.json', cfg)
    # The supervised review waits for atomic producer outputs; it never starts a competing encoder.
    while not all((model / d / f'held_fold{f}_scores.npz').is_file() for d in cases for f in range(4)) or not (model / 'summary.json').is_file():
        budget(); print('waiting for original producer held scores', flush=True); time.sleep(5)
    STATE['selected'] = select()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool: results = list(pool.map(replay, cases))
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool: records = list(pool.map(render, cases))
    records = sorted([r for group in records for r in group], key=lambda r: r['identity']['audit_index'])
    write(out / 'figure_manifest.json', dict(status='rendered_pending_actual_open', examples=records, figure_count=2 * len(records)))
    summary = json.loads((model / 'summary.json').read_text())
    write(out / 'independent_replay.json', dict(status='passed', arms=list(ARMS), datasets=results, crops=400, position_windows=17040,
        all_primary_and_dynamic_metric_arrays_bitwise_exact=True, independent14bit_pack_unpack=True, independent_paid_atom_lookup=True,
        decoder_inputs='actual saved integers, frozen dictionaries, frozen posterior state, fixed native order',
        decoder_encoder_or_source_calls=0, validation_reads=0, confirmation_reads=0, test_reads=0,
        owner_summary=summary, code_sha256=sha(Path(__file__)), figures_pending_actual_open=2 * len(records)))


if __name__ == '__main__': main()
