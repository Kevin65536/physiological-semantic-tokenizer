#!/usr/bin/env python3
"""Train-only signal distortion controls for the unified-grid tokenizer audit."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from scipy.signal import periodogram

from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.data.dual_component_tokenizer import base_tokenizer_batch

DEFAULT_OUTPUT = ROOT / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1/calibration'
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def stable_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite(v) for v in value]
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plan_examples(rows, modality):
    """Metadata-only bounded sample. Selection never sees signal values."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        if row['split'] == 'train' and row['modality'] == modality and row['inputs']['score']:
            groups[row['dataset']][row['subject']][row['record']].append(index)
    examples = []
    for dataset, subjects in sorted(groups.items()):
        selected_subjects = sorted(subjects, key=lambda s: stable_key([20261123, dataset, s]))[:8]
        for subject in selected_subjects:
            records = subjects[subject]
            selected_records = sorted(records, key=lambda r: stable_key([20261123, dataset, subject, r]))[:2]
            for record in selected_records:
                index = min(records[record], key=lambda i: stable_key([20261123, rows[i]['parent']]))
                crops = sorted(rows[index]['inputs']['score'], key=lambda item: item['start_s'])
                crop_indices = sorted(set((0, len(crops)-1)))
                for crop_index in crop_indices:
                    crop = crops[crop_index]
                    positions = crop['positions']
                    for p in sorted(set((positions[0], positions[len(positions)//2], positions[-1]))):
                        examples.append((index, crop['start_s'], p))
    return examples


def correlation(y, p, axis=1):
    y = y-y.mean(axis=axis, keepdims=True)
    p = p-p.mean(axis=axis, keepdims=True)
    denominator = np.sqrt(np.sum(y*y, axis=axis)*np.sum(p*p, axis=axis))
    return np.divide(np.sum(y*p, axis=axis), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 1e-20)


def band_projection(y, rate, lo, hi):
    f = np.fft.rfftfreq(y.shape[1], 1/rate)
    z = np.fft.rfft(y, axis=1)
    z[:, ~((f >= lo) & (f < hi)), :] = 0
    return np.fft.irfft(z, n=y.shape[1], axis=1)


def distortions(y, rate, modality):
    yield 'identity', y.copy()
    yield 'gain_0.95', 0.95*y
    yield 'gain_0.8', 0.8*y
    yield 'gain_0.5', 0.5*y
    yield 'offset_0.1_input_SD', y+0.1
    patch = y.reshape(len(y), 15, -1, y.shape[-1])
    means = patch.mean(2, keepdims=True)
    yield 'patch_mean_only', np.broadcast_to(means, patch.shape).reshape(y.shape).copy()
    yield 'patch_demeaned', (patch-means).reshape(y.shape)
    for delay_s in ((0.005, 0.025, 0.05) if modality == 'eeg' else (0.1, 0.5, 1.0)):
        lag = round(delay_s*rate)
        shifted = np.concatenate((np.repeat(y[:, :1], lag, axis=1), y[:, :-lag]), axis=1)
        yield f'delay_{delay_s:g}s', shifted
    if modality == 'eeg':
        for hi in (8, 13, 30):
            yield f'lowpass_{hi}Hz', band_projection(y, rate, 0, hi)
        yield 'remove_alpha_8_13Hz', y-band_projection(y, rate, 8, 13)
        yield 'remove_theta_4_8Hz', y-band_projection(y, rate, 4, 8)
    else:
        centered_time = np.arange(patch.shape[2], dtype=float)
        centered_time -= centered_time.mean()
        slopes = np.sum(patch*centered_time[None, None, :, None], axis=2, keepdims=True)/np.sum(centered_time**2)
        linear = means+slopes*centered_time[None, None, :, None]
        yield 'patch_linear_only', linear.reshape(y.shape)
        yield 'patch_half_slope', (patch-0.5*slopes*centered_time[None, None, :, None]).reshape(y.shape)


def control_metrics(y, p, variance, difference_variance, scales, modality):
    error = p-y
    result = dict(nmse=np.mean(error**2, axis=1)/variance,
                  rmse_native=np.sqrt(np.mean(error**2, axis=1))*scales[:, None],
                  correlation=correlation(y, p), bias_native=error.mean(1)*scales[:, None],
                  rms_gain=np.sqrt(np.mean(p*p, axis=1)/np.maximum(np.mean(y*y, axis=1), 1e-20)))
    result['absolute_rms_gain_error'] = abs(result['rms_gain']-1)
    yp = y.reshape(len(y), 15, -1, y.shape[-1])
    pp = p.reshape(yp.shape)
    result['local_rms_correlation'] = correlation(np.sqrt(np.mean(yp*yp, axis=2)), np.sqrt(np.mean(pp*pp, axis=2)))
    result['local_rms_nrmse'] = np.sqrt(np.mean((np.sqrt(np.mean(yp*yp, axis=2))-np.sqrt(np.mean(pp*pp, axis=2)))**2, axis=1)/variance)
    boundaries = np.arange(1, 15)*(y.shape[1]//15)
    edge_error = error[:, boundaries]-error[:, boundaries-1]
    result['boundary_jump_difference_nmse'] = np.mean(edge_error**2, axis=1)/difference_variance
    if modality == 'eeg':
        f, yp_power = periodogram(y, fs=200, window='hann', axis=1)
        _, pp_power = periodogram(p, fs=200, window='hann', axis=1)
        for lo, hi in BANDS:
            keep = (f >= lo) & (f < hi)
            result[f'band_{lo}_{hi}_log10_ratio'] = np.log10(np.maximum(pp_power[:, keep].sum(1), 1e-20)/np.maximum(yp_power[:, keep].sum(1), 1e-20))
            result[f'band_{lo}_{hi}_absolute_log10_ratio'] = abs(result[f'band_{lo}_{hi}_log10_ratio'])
    else:
        dt = 0.1
        for order, label in ((1, 'slope'), (2, 'curvature')):
            yd = np.diff(y, n=order, axis=1)/(dt**order)
            pd = np.diff(p, n=order, axis=1)/(dt**order)
            result[label+'_correlation'] = correlation(yd, pd)
            result[label+'_rms_ratio'] = np.sqrt(np.mean(pd*pd, axis=1)/np.maximum(np.mean(yd*yd, axis=1), 1e-20))
            result[label+'_nrmse_local'] = np.sqrt(np.mean((yd-pd)**2, axis=1)/np.maximum(np.mean(yd*yd, axis=1), 1e-20))
            # Compute each derivative inside its own token interval, preventing
            # boundary jumps from masquerading as preserved local dynamics.
            y_inside = np.diff(yp, n=order, axis=2).reshape(len(y), -1, y.shape[-1])/(dt**order)
            p_inside = np.diff(pp, n=order, axis=2).reshape(len(y), -1, y.shape[-1])/(dt**order)
            result['interior_'+label+'_correlation'] = correlation(y_inside, p_inside)
            result['interior_'+label+'_nrmse_local'] = np.sqrt(np.mean((y_inside-p_inside)**2, axis=1)/np.maximum(np.mean(y_inside*y_inside, axis=1), 1e-20))
            result['interior_'+label+'_rms_ratio'] = np.sqrt(np.mean(p_inside*p_inside, axis=1)/np.maximum(np.mean(y_inside*y_inside, axis=1), 1e-20))
        y_shape = (yp-yp.mean(2, keepdims=True)).reshape(y.shape)
        p_shape = (pp-pp.mean(2, keepdims=True)).reshape(y.shape)
        result['demeaned_patch_shape_correlation'] = correlation(y_shape, p_shape)
        result['peak_time_error_s'] = abs(np.argmax(p, axis=1)-np.argmax(y, axis=1))*dt
        result['trough_time_error_s'] = abs(np.argmin(p, axis=1)-np.argmin(y, axis=1))*dt
    return result


def aggregate(values, examples, rows):
    """Positions/crops -> record -> subject -> dataset. Keep distributions too."""
    subjects = defaultdict(lambda: defaultdict(list))
    for value, (index, _, _) in zip(values, examples):
        row = rows[index]
        subjects[row['dataset'], row['subject']][row['record']].append(value)
    by_subject = []
    for (dataset, subject), records in sorted(subjects.items()):
        a = np.asarray([np.nanmean(v, axis=0) for v in records.values()])
        by_subject.append(dict(dataset=dataset, subject=subject, value=np.nanmean(a, axis=0)))
    datasets = defaultdict(list)
    for item in by_subject:
        datasets[item['dataset']].append(item['value'])
    by_dataset = {dataset: np.nanmean(a, axis=0) for dataset, a in sorted(datasets.items())}
    return finite(dict(macro=np.nanmean(list(by_dataset.values()), axis=0), by_dataset=by_dataset,
                       by_subject=by_subject, sample_p10_p50_p90=np.nanquantile(values, (.1, .5, .9), axis=0)))


def gaussian_rd(eigenvalues, bits):
    """Reverse-waterfilling diagnostic under an empirical Gaussian patch model."""
    lam = eigenvalues[eigenvalues > eigenvalues.max()*1e-12]
    lower, upper = lam.min()*1e-12, lam.max()
    for _ in range(100):
        theta = np.sqrt(lower*upper)
        rate = 0.5*np.log2(np.maximum(lam/theta, 1)).sum()
        if rate > bits:
            lower = theta
        else:
            upper = theta
    return float(np.minimum(lam, upper).sum()/lam.sum())


def render_controls(output, rows, scales, plans):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': font_name, 'axes.unicode_minus': False})
    cache = OrderedDict()
    figures = []
    for modality, examples in plans.items():
        chosen = []
        for example in examples:
            index, start, p = example
            if rows[index]['dataset'] in {rows[i]['dataset'] for i, _, _ in chosen}:
                continue
            batch = base_tokenizer_batch(rows, [example], scales, array_cache=cache, cache_limit=32)
            if batch['mask'].all():
                chosen.append(example)
        fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 3*len(chosen)), squeeze=False)
        for axes_row, example in zip(axes, chosen):
            index, start, p = example
            row = rows[index]
            batch = base_tokenizer_batch(rows, [example], scales, array_cache=cache, cache_limit=32)
            rate = row['rate']; channels = row['channels']
            scale = scales['positions'][modality][row['dataset']][row['positions'][p]]['scale']
            y = batch['s'].reshape(1, 30*rate, channels)
            label = 'remove_alpha_8_13Hz' if modality == 'eeg' else 'patch_mean_only'
            p_values = dict(distortions(y, rate, modality))[label]
            t = np.arange(30*rate)/rate
            for axis, limits in zip(axes_row, ((0, 30), (12, 14.5))):
                for channel in range(channels):
                    colour = ('#245a9c', '#a94d3b')[channel]
                    name = 'S电压' if modality == 'eeg' else ('S HbO', 'S HbR')[channel]
                    axis.plot(t, y[0, :, channel]*scale, color=colour, linewidth=.85, label=name+'目标')
                    axis.plot(t, p_values[0, :, channel]*scale, color=colour, linestyle='--', linewidth=.85, alpha=.85, label=name+'失真对照')
                for boundary in np.arange(2, 30, 2):
                    axis.axvline(boundary, color='0.75', linewidth=.4)
                axis.set_xlim(*limits)
                axis.set_ylabel(row['source_unit'])
                axis.set_xlabel('时间 / s')
                axis.legend(loc='upper right', fontsize=7, ncol=channels)
            axes_row[0].set_title(row['dataset']+' / '+row['subject']+' / '+row['positions'][p], fontsize=10)
            axes_row[1].set_title(('抹去8–13 Hz' if modality == 'eeg' else '每2秒以均值替代')+'，同一纵轴坐标', fontsize=10)
            axes_row[1].set_ylim(axes_row[0].get_ylim())
        fig.suptitle('训练集失真校准，'+('EEG-S' if modality == 'eeg' else 'Hb-S')+'，灰线为2秒token边界。每库按固定元数据顺序首个完整支持样本。', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .97))
        path = output/(modality+'_distortion_examples.png')
        fig.savefig(path, dpi=220)
        plt.close(fig)
        figures.append(dict(path=str(path), selection=[dict(dataset=rows[i]['dataset'], subject=rows[i]['subject'], parent=rows[i]['parent'], start_s=start, position=rows[i]['positions'][p]) for i, start, p in chosen]))
    write_json(output/'figures.json', figures)


def matched_audit_controls(output, rows, scales, plans):
    """Calibrate the exact candidate metric owner on all planned scoring masks."""
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
    destination = output/'matched_metric_controls.json'
    if destination.exists():
        raise ValueError('matched control evidence already exists')
    results = {}
    cache = OrderedDict()
    for modality, examples in plans.items():
        batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=64)
        channels = 1 if modality == 'eeg' else 2
        rate = 200 if modality == 'eeg' else 10
        shape = (len(examples), 30*rate, channels)
        input_scales = np.asarray([scales['positions'][modality][rows[i]['dataset']][rows[i]['positions'][p]]['scale'] for i, _, p in examples])
        streams = [distortions(batch[key].reshape(shape), rate, modality) for key in ('x', 's', 'o')]
        for controls in zip(*streams):
            names = [name for name, _ in controls]
            if len(set(names)) != 1:
                raise ValueError('distortion controls must align across components')
            predictions = {key: value.reshape(batch['x'].shape) for key, (_, value) in zip(('sum', 'S', 'O'), controls)}
            metrics = joint_reconstruction_metrics(predictions, batch, channels, input_scales)
            results[modality+'/'+names[0]] = joint_reconstruction_summary(metrics['metrics'], metrics['support'], examples, rows)
    write_json(destination, finite(dict(schema='tokenizer_distortion_matched_metrics_v1',
                role='train-only metric calibration; each X/S/O target independently transformed by the named fixed operator',
                candidate_metric_owner='src.analysis.dual_component_tokenizer.joint_reconstruction_metrics',
                partial_support='all planned windows included with actual scoring masks', controls=results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--matched-metrics', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists() and not (args.render_only or args.matched_metrics):
        raise ValueError('completed calibration is immutable; use a fresh output')
    started = time.monotonic()
    cfg = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    plans = {m: plan_examples(rows, m) for m in ('eeg', 'hb')}
    if args.matched_metrics:
        matched_audit_controls(args.output, rows, scales, plans)
        return
    if args.render_only:
        if not (args.output/'summary.json').exists():
            raise ValueError('render requires completed controls')
        render_controls(args.output, rows, scales, plans)
        return
    identities = {}
    for modality, examples in plans.items():
        identities[modality] = [dict(dataset=rows[i]['dataset'], subject=rows[i]['subject'], split=rows[i]['split'],
                                    record=rows[i]['record'], parent=rows[i]['parent'], start_s=start,
                                    position=rows[i]['positions'][p], row_index=i, position_index=p)
                                for i, start, p in examples]
    write_json(args.output/'plan.json', dict(schema='tokenizer_distortion_calibration_v1',
               created=datetime.now(timezone.utc).isoformat(), source=cfg['data'],
               plan_identity=stable_key(identities), selection='SHA256(seed,dataset,subject), then record, then one parent per record; first/last score crop and first/middle/last available position',
               caps=dict(subjects_per_dataset=8, records_per_subject=2, parents_per_record=1, crops_per_parent=2, positions_per_crop=3),
               fitting_split='train', examples=identities))
    controls = {}; dimensions = {}; closure = {}; spectra = {}; numerical = {}
    cache = OrderedDict()
    for modality, examples in plans.items():
        batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=64)
        channels = 1 if modality == 'eeg' else 2
        rate = 200 if modality == 'eeg' else 10
        data = {k: np.asarray(batch[k], float).reshape(len(examples), 30*rate, channels) for k in ('x', 's', 'o')}
        mask = np.asarray(batch['mask'], bool).reshape(data['x'].shape)
        valid = mask.all(axis=(1, 2))
        valid_examples = [example for example, ok in zip(examples, valid) if ok]
        numerical[modality] = dict(planned_inputs=len(examples), full_support_inputs=int(valid.sum()),
                                   partial_support_inputs=int((~valid).sum()), full_support_fraction=float(valid.mean()))
        differences = data['s']+data['o']-data['x']
        closure[modality] = dict(max_abs_train_scaled=float(np.max(np.abs(differences[mask]))),
                                rms_train_scaled=float(np.sqrt(np.mean(differences[mask]**2))))
        scale_values = np.asarray([scales['positions'][modality][rows[i]['dataset']][rows[i]['positions'][p]]['scale'] for i, _, p in examples])[valid]
        for branch in ('x', 's', 'o'):
            y = data[branch][valid]
            variance = np.asarray(batch['v'+branch])[:, 0, :channels][valid]
            difference_variance = np.asarray(batch['dv'+branch])[:, 0, :channels][valid]
            for label, p in distortions(y, rate, modality):
                values = control_metrics(y, p, variance, difference_variance, scale_values, modality)
                controls[modality+'/'+branch+'/'+label] = {k: aggregate(v, valid_examples, rows) for k, v in values.items()}
            for dataset in sorted({rows[i]['dataset'] for i, _, _ in valid_examples}):
                keep = np.asarray([rows[i]['dataset'] == dataset for i, _, _ in valid_examples])
                patches = y[keep].reshape(-1, 2*rate*channels)
                centered = patches-patches.mean(0)
                cov = centered.T@centered/max(len(patches)-1, 1)
                lam = np.maximum(np.linalg.eigvalsh(cov), 0)[::-1]
                fractions = lam/lam.sum()
                dimensions[modality+'/'+branch+'/'+dataset] = dict(patches=len(patches), vector_dimensions=patches.shape[1],
                    participation_ratio=float(lam.sum()**2/(lam@lam)), entropy_effective_rank=float(np.exp(-np.sum(fractions[fractions>0]*np.log(fractions[fractions>0])))),
                    pca90=int(np.searchsorted(np.cumsum(fractions), .9)+1), pca95=int(np.searchsorted(np.cumsum(fractions), .95)+1),
                    gaussian_assumption_rd_nmse={str(bits):gaussian_rd(lam,bits) for bits in (7, 10, 12, 16, 20, 32, 64)},
                    eigenvalues=lam.tolist())
                if modality == 'eeg':
                    frequencies, power = periodogram(y[keep], fs=200, window='hann', axis=1)
                    total = power.sum(axis=1)
                    values = {}
                    for lo, hi in BANDS:
                        values[f'{lo}_{hi}Hz'] = np.mean(power[:, (frequencies>=lo)&(frequencies<hi)].sum(axis=1)/np.maximum(total, 1e-20), axis=0).tolist()
                    values['outside_1_45Hz'] = np.mean(power[:, (frequencies<1)|(frequencies>=45)].sum(axis=1)/np.maximum(total, 1e-20), axis=0).tolist()
                    spectra[modality+'/'+branch+'/'+dataset] = values
    summary = dict(schema='tokenizer_distortion_calibration_v1', status='completed',
                   elapsed_seconds=time.monotonic()-started, numerical=numerical, closure=closure,
                   controls=controls, effective_dimensions=dimensions, target_spectral_energy=spectra,
                   definitions=dict(nmse='mean squared error / G0 frozen training variance separately for X,S,O and Hb color',
                       rmse_native='sqrt(mean squared train-scaled error) multiplied by exact G0 train X position SD',
                       local_rms='RMS within each nonoverlapping 2s token interval; temporal correlation across 15 intervals',
                       boundary='squared reconstructed-minus-target adjacent-sample boundary jump / frozen training adjacent-difference variance',
                       dimension='empirical covariance of 2s train patches in the fixed G0 position scale, centered across patches',
                       rate_distortion='reverse waterfilling under empirical Gaussian vector source and quadratic distortion; diagnostic assumption only, estimated from dependent train patches, lacks operational single-ID achievability or an impossibility guarantee',
                       masks='closure uses actual joint mask. Distortion/dimension comparisons require complete30s joint support; all partial windows retained in counts. No validation or regression arrays read.'))
    write_json(args.output/'summary.json', finite(summary))
    print(json.dumps(dict(status='completed', elapsed_seconds=summary['elapsed_seconds'], numerical=numerical, closure=closure)))


if __name__ == '__main__':
    main()
