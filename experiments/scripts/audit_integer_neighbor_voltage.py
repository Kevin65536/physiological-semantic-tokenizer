"""Replay full integer montages and test whether neighboring voltage retains events."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import fcntl
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
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_joint_eeg_context
from src.tokenizers.spatial_waveform_dictionary import load_model

STATE = {}


def write(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(cfg):
    unit = os.environ.get('TOKENIZER_INTEGER_NEIGHBOR_UNIT')
    used = time.process_time() if not unit else int(subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'CPUUsageNSec', '--value'], text=True))/1e9
    if float(os.environ.get('TOKENIZER_INTEGER_NEIGHBOR_PRIOR_CPU', '0'))+used >= cfg['resources']['stop_CPU_seconds']:
        raise RuntimeError('bounded integer-neighbor CPU budget reached')
    return used


def project(values):
    values = np.asarray(values, float)
    if values.ndim != 3 or values.shape[:2] != (15, 400):
        raise ValueError('fifteen native 400-sample patches required')
    hz = np.fft.rfftfreq(400, 1/200)
    spectrum = np.fft.rfft(values, axis=1)
    spectrum[:, ~((hz >= 1) & (hz <= 45))] = 0
    return np.fft.irfft(spectrum, n=400, axis=1)


def predict_patch(design, target, complete, test, ridge=.01):
    design, target, complete = np.asarray(design, float), np.asarray(target, float), np.asarray(complete, bool)
    if design.ndim != 3 or design.shape[:2] != (15, 400) or target.shape != (15, 400) or complete.shape != (15,) or not complete[test]:
        raise ValueError('same full context, target and complete held patch required')
    fit = complete.copy();fit[test] = False
    if not fit.any() or design.shape[-1] == 0:
        raise ValueError('other complete fitting patches and predictors required')
    matrix, labels = design[fit].reshape(-1, design.shape[-1]), target[fit].reshape(-1)
    scaling = np.sqrt(np.maximum(np.mean(matrix**2, axis=0), 1e-12))
    z, held = matrix/scaling, design[test]/scaling
    gram = z.T@z/len(z)
    coefficient = np.linalg.solve(gram+ridge*np.eye(len(scaling)), z.T@labels/len(z))
    return held@coefficient, dict(coefficients=coefficient, scales=scaling, fit_patches=int(fit.sum()),
        same_patch_oracle=held@np.linalg.lstsq(held, target[test], rcond=1e-8)[0])


def describe(target, prediction, scale):
    target, prediction = np.asarray(target), np.asarray(prediction)
    energy = float(np.mean(target**2))
    rms_y, rms_p = np.sqrt(energy), np.sqrt(np.mean(prediction**2))
    envelope_y = np.sqrt(np.mean(target.reshape(8, 50)**2, axis=1))
    envelope_p = np.sqrt(np.mean(prediction.reshape(8, 50)**2, axis=1))
    def correlation(y, p):
        return float(np.corrcoef(y, p)[0, 1]) if min(np.std(y), np.std(p)) > 0 else None
    return dict(error_over_target_energy=float(np.mean((prediction-target)**2)/energy),
        RMSE_native=float(np.sqrt(np.mean((prediction-target)**2))*scale),
        RMS_gain=float(rms_p/rms_y), correlation=correlation(target, prediction),
        bias_native=float(np.mean(prediction-target)*scale),
        absolute_voltage_peak_delay_s=float((np.argmax(abs(prediction))-np.argmax(abs(target)))/200),
        RMS_peak_delay_s=float((np.argmax(envelope_p)-np.argmax(envelope_y))*.25),
        RMS_envelope_correlation=correlation(envelope_y, envelope_p),
        RMS_envelope_NRMSE=float(np.sqrt(np.mean((envelope_p-envelope_y)**2))/rms_y))


def admission(records, cfg):
    rule = cfg['admission'];low, high = rule['RMS_gain_interval']
    recovered, adverse = [], []
    for item in records:
        if 'arms' not in item:
            adverse.append(dict(index=item['index'], reason='failed_or_missing'));continue
        baseline = item['arms']['original']['error_over_target_energy']
        actual = item['arms']['integer_neighbors']
        ratio = actual['error_over_target_energy']/baseline
        if ratio > rule['all_17_error_ratio_to_original_max']:
            adverse.append(dict(index=item['index'], dataset=item['dataset'], ratio=ratio))
        if ('candidate_max' in item['selection'] and actual['error_over_target_energy'] <= rule['integer_neighbor_error_over_target_energy_max']
                and ratio <= rule['integer_neighbor_error_ratio_to_original_max'] and low <= actual['RMS_gain'] <= high
                and abs(actual['RMS_peak_delay_s']) <= rule['RMS_peak_delay_absolute_max_s']):
            recovered.append(item['dataset'])
    return dict(admitted=len(set(recovered)) >= rule['required_candidate_max_datasets'] and not adverse and len(records) == 17,
        recovered_candidate_max_datasets=sorted(set(recovered)), adverse_examples=adverse,
        original_examples=len(records), rule=rule, controls_used_for_admission=False)


def dataset_worker(dataset):
    cfg, run, rows, scales, examples, frozen, folds, out = (STATE[k] for k in ('cfg', 'run', 'rows', 'scales', 'examples', 'frozen', 'folds', 'out'))
    original = json.loads((run/cfg['source']['gallery']/'neighbor_voltage_diagnostic.json').read_text())['cases']
    stored = {}
    for path in sorted((run/cfg['source']['actual_IDs']/dataset).glob('ids_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            for j, key in enumerate(z['keys']):
                stored[tuple(map(int, key))] = {b: z['transient_'+b][j:j+1].copy() for b in ('S', 'O')}
    models, decoded, result = {}, {}, []
    for index, item in enumerate(examples):
        if item['dataset'] != dataset: continue
        budget(cfg)
        row = rows[item['row']]
        if (row['split'], row['modality'], row['subject'], row['parent']) != ('train', 'eeg', item['subject'], item['parent']):
            raise ValueError('original displayed training identity required before arrays')
        fold = folds[dataset][row['subject']]
        if fold not in models:
            path = run/cfg['source']['frozen_decoders']/f'{dataset}_fold{fold}.npz'
            models[fold] = load_model(path)
            estimator = models[fold].estimator
            if estimator.metadata['held_fold'] != fold or row['subject'] in estimator.metadata['fit_subjects']:
                raise ValueError('frozen decoder must exclude the displayed subject')
        model = models[fold]
        names = model.metadata['positions'][dataset]
        lookup = {name.upper(): j for j, name in enumerate(names)}
        order = np.asarray([lookup[name.upper()] for name in row['positions']])
        key = item['row'], item['start_s'], fold
        ids = stored[key]
        if key not in decoded:
            # There is no X argument on this path: saved S/O integers and the
            # matching frozen decoder determine the entire native waveform.
            wave = model.decode_ids(ids, dataset)
            again = load_model(run/cfg['source']['frozen_decoders']/f'{dataset}_fold{fold}.npz').decode_ids(ids, dataset)
            for branch in ('S', 'O', 'sum'):
                np.testing.assert_array_equal(wave[branch], again[branch])
            decoded[key] = wave
        wave = decoded[key]
        native = lookup[item['position_name'].upper()]
        for branch in ('S', 'O'):
            np.testing.assert_array_equal(ids[branch][0, native], frozen['ids_'+branch][index])
            np.testing.assert_array_equal(wave[branch][0, native], frozen['prediction_'+branch][index])
        np.testing.assert_array_equal(wave['sum'][0, native], frozen['prediction_sum'][index])
        crop = dict(row=item['row'], start_s=item['start_s'], positions=[item['position']])
        x, seen, _ = base_tokenizer_joint_eeg_context(rows, crop, scales, row['positions'], allowed_splits=('train',))
        np.testing.assert_array_equal(seen, np.any(ids['S'][0, order] != 0, axis=1))
        np.testing.assert_array_equal(seen, np.any(ids['O'][0, order] != 0, axis=1))
        site = item['position'];complete = frozen['mask'][index].astype(bool).all(-1)
        raw = x.reshape(15, 400, -1)
        np.testing.assert_array_equal(raw[complete, :, site], frozen['x'][index, complete])
        signal = project(raw)
        integer = project(wave['sum'][0, order].transpose(1, 2, 0))
        candidate = integer[:, :, site]
        y = signal[:, :, site]
        errors = np.where(complete, np.mean((candidate-y)**2, axis=1), -1)
        test = int(errors.argmax())
        if not seen[site] or test != original[index]['worst_complete_patch']:
            raise ValueError('original target availability or worst complete patch changed')
        channels = np.flatnonzero(seen & (np.arange(len(seen)) != site))
        designs = dict(raw_neighbors=signal[:, :, channels], integer_neighbors=integer[:, :, channels],
            integer_all=integer[:, :, seen], integer_self=integer[:, :, site:site+1])
        estimates, fits, arms = {}, {}, {'original': describe(y[test], candidate[test], item['scale'])}
        for arm, design in designs.items():
            prediction, fit = predict_patch(design, y, complete, test, cfg['diagnostic']['ridge'])
            estimates[arm] = prediction;fits[arm] = {k: v for k, v in fit.items() if k != 'same_patch_oracle'}
            arms[arm] = describe(y[test], prediction, item['scale'])
            arms[arm+'_same_patch_oracle'] = describe(y[test], fit['same_patch_oracle'], item['scale'])
        np.testing.assert_allclose(arms['original']['error_over_target_energy'], original[index]['candidate_error_over_target_energy'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arms['raw_neighbors']['error_over_target_energy'], original[index]['other_electrodes_ridge_error_ratio'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arms['raw_neighbors_same_patch_oracle']['error_over_target_energy'], original[index]['other_electrodes_same_patch_oracle_ratio'], rtol=1e-10, atol=1e-12)
        path = out/f'case_{index:02d}.npz'
        payload = dict(ids_S=ids['S'], ids_O=ids['O'], full_integer_sum=wave['sum'],
            raw_target_full=raw[:, :, site], original_prediction_full=wave['sum'][0, native], complete=complete,
            target_patch=y[test], original_patch=candidate[test], **{arm+'_patch': value for arm, value in estimates.items()})
        if STATE.get('review'):
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(payload): raise ValueError('saved diagnostic array interface changed')
                for name, value in payload.items(): np.testing.assert_array_equal(saved[name], value)
        else:
            np.savez_compressed(path, **payload)
        result.append(dict(index=index, dataset=dataset, subject=item['subject'], site=item['position_name'],
            row=item['row'], parent=item['parent'], start_s=item['start_s'], selection=item['selection'],
            source_fold=fold, worst_complete_patch=test, fit_patches=int(complete.sum()-1), neighbors=len(channels),
            integer_model=str(run/cfg['source']['frozen_decoders']/f'{dataset}_fold{fold}.npz'),
            full_montage_saved_integer_replay_exact=True, gallery_arrays_and_original_raw_control_reproduced=True,
            units=item['source_unit'], scale=item['scale'], arms=arms, fits=fits))
    return result


def render(records, out, version=1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(original='#bd6d27', raw_neighbors='#61799d', integer_neighbors='#00847f', integer_all='#9c5079', integer_self='#808080')
    paths = []
    for record in records:
        i, test, scale = record['index'], record['worst_complete_patch'], record['scale']
        with np.load(out/f'case_{i:02d}.npz', allow_pickle=False) as a:
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), layout='constrained')
            time_grid = np.arange(6000)/200
            axes[0].plot(time_grid, a['raw_target_full'].reshape(-1)*scale, color='#213c51', lw=.7, label='Observed target X')
            axes[0].plot(time_grid, a['original_prediction_full'].reshape(-1)*scale, color=colors['original'], lw=.7, label='Original integer X')
            for edge in np.arange(2, 30, 2): axes[0].axvline(edge, color='gray', ls=':', lw=.4)
            for patch in np.flatnonzero(~a['complete']): axes[0].axvspan(patch*2, patch*2+2, color='gray', alpha=.2)
            axes[0].axvspan(test*2, test*2+2, color='#f1cb91', alpha=.3)
            axes[0].set(xlim=(0, 30), xlabel='Crop time (s)', ylabel=record['units'], title='Original 30-s voltage; selected patch shaded')
            clock = test*2+np.arange(400)/200
            axes[1].plot(clock, a['target_patch']*scale, color='#213c51', lw=1, label='Target 1–45 Hz')
            rms_clock = test*2+(np.arange(8)+.5)*.25
            axes[2].plot(rms_clock, np.sqrt(np.mean(a['target_patch'].reshape(8, 50)**2, axis=1))*scale, color='#213c51', marker='.', label='Target')
            for arm in colors:
                signal = a['original_patch' if arm == 'original' else arm+'_patch']
                axes[1].plot(clock, signal*scale, color=colors[arm], lw=.85, label=arm.replace('_', ' '))
                axes[2].plot(rms_clock, np.sqrt(np.mean(signal.reshape(8, 50)**2, axis=1))*scale, color=colors[arm], marker='.', label=arm.replace('_', ' '))
            axes[1].set(xlim=(test*2, test*2+2), xlabel='Crop time (s)', ylabel=record['units'], title='Same selected 2 s, inclusive 1–45 Hz')
            axes[2].set(xlim=(test*2, test*2+2), xlabel='Crop time (s)', ylabel='0.25-s RMS | '+record['units'], title='Held-patch amplitude and event timing')
            axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);axes[2].legend(fontsize=7)
            fig.suptitle(f"{i}: {record['dataset']} / {record['subject']} / {record['site']} / +{record['start_s']} s / fold {record['source_fold']}\nOnly other complete patches fit ridge=.01; target-assisted mechanism diagnostic. Same-patch oracle omitted from curves.", fontsize=10)
            fig.supxlabel('Dotted lines: 2-s boundaries; gray: incomplete scoring patch excluded from fitting. All curves retain the same native coordinate.', fontsize=8)
            suffix = '' if version == 1 else f'_v{version}'
            path = out/f'neighbor_event_{i:02d}{suffix}.png'
            if path.exists(): raise FileExistsError('retained scientific figure version')
            fig.savefig(path, dpi=180, bbox_inches='tight');plt.close(fig);paths.append(str(path))
    write(out/('figure_manifest.json' if version == 1 else f'figure_manifest_v{version}.json'), dict(figures=paths, actual_opening_required=True, source='case_*.npz and summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True);parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', action='store_true')
    args = parser.parse_args();cfg=yaml.safe_load(args.config.read_text());out=args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'summary.json').exists() and not args.review: raise FileExistsError('retained integer-neighbor diagnostic')
    if cfg['data']['allowed_splits'] != ['train'] or any(cfg['data'][k] != 'forbidden' for k in ('validation_access','confirmation_access','test_access')):
        raise ValueError('original training-only scope required before arrays')
    run=ROOT/cfg['source']['run'];gallery=run/cfg['source']['gallery']
    manifest, scales, _, _=base_tokenizer_training_data(yaml.safe_load((ROOT/cfg['source']['g0']).read_text()), ROOT)
    examples=json.loads((gallery/'waveform_examples.json').read_text())['examples']
    if len(examples) != 17: raise ValueError('all original seventeen display windows required')
    folds=json.loads((run/cfg['source']['folds']).read_text())
    with np.load(gallery/'waves.npz', allow_pickle=False) as z: frozen={k:z[k].copy() for k in z.files}
    if args.review:
        if json.loads((out/'resolved_config.json').read_text()) != cfg: raise ValueError('frozen diagnostic rules changed')
    else: write(out/'resolved_config.json', cfg)
    STATE.update(cfg=cfg, run=run, rows=manifest['rows'], scales=scales, examples=examples, frozen=frozen, folds=folds, out=out, review=args.review)
    with threadpool_limits(1):
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
            result=sorted(sum(pool.map(dataset_worker, sorted({r['dataset'] for r in examples})), []), key=lambda r:r['index'])
        decision=admission(result, cfg)
        if args.review:
            def convert(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, np.generic): return obj.item()
                raise TypeError(type(obj).__name__)
            previous=json.loads((out/'summary.json').read_text())
            if json.loads(json.dumps(result,default=convert)) != previous['cases'] or decision != previous['admission']:
                raise AssertionError('independent full-integer replay, control or decision changed')
            write(out/'independent_review.json',dict(cases=17,all_frozen_integer_arrays_predictions_and_ridge_metrics_exact=True,
                original_control_and_worst_patch_reproduced=True,admission_unchanged=True,CPU_seconds=budget(cfg)))
            render(result,out,version=2)
            return
        write(out/'summary.json', dict(status='completed_diagnostic', cases=result, admission=decision,
            formal_fitting_started=False, validation_reads=0, confirmation_reads=0, test_reads=0,
            cohort='same seventeen displayed train cases; target-assisted within-window diagnostic',
            original_script_sha256=sha(gallery/'audit_neighbor_voltage.py'), CPU_seconds=budget(cfg)))
        render(result, out)


if __name__ == '__main__': main()
