#!/usr/bin/env python3
"""Versioned public observation-semantics suite; all stages are resumable.

A reads named retained evidence. B/C read registered native public records
through the unified loader, after software checks. D is explicitly synthetic.
Run substantial stages under the project's durable supervisor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, resample_poly
import yaml

from src.inference.observation_semantics import (
    lagged, event_impulses, coordinate_stability, hardware_donors,
    block_interval, sign_flip_p, holm, candidate_inference,
)
from src.inference.shared_driver_attribution import (
    hb_diagnostic_coordinates, fit_standardized_ridge, predict_standardized_ridge,
)
from src.inference.observation_baselines import native_feature_operators, eeg_band_power

DEFAULT_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/observation_semantics_v1.yaml'


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema'] != 'observation_semantics_v1' or cfg['protected_data'] != 'forbidden':
        raise ValueError('only the public versioned observation-semantics contract is supported')
    if cfg['cache_root'] != 'data/cache/physiology_semantic_clean_v5':
        raise ValueError('native reads require the specified v5 public cache identity')
    if cfg['tensor']['hb_steps'] != 120 or cfg['tensor']['hb_order'] != ['HbO', 'HbR']:
        raise ValueError('unexpected tensor contract')
    if cfg['measured']['split'] != 'odd_native_subject_number_train_even_evaluate':
        raise ValueError('unsupported split')
    if cfg['mechanisms']['steps'] != 120 or cfg['mechanisms']['candidates_per_mechanism'] != 9:
        raise ValueError('unexpected finite mechanism panel')
    return cfg


def phase(subject):
    return 'train' if int(re.search(r'(\d+)$', subject).group(1)) % 2 else 'evaluate'


def nonoverlap(refs, limit):
    chosen = []
    for r in sorted(refs, key=lambda v: (v['eeg_start_s'], v['hb_start_s'], v['id'])):
        if any(abs(r['eeg_start_s']-x['eeg_start_s']) < 30.-1e-6 or
               abs(r['hb_start_s']-x['hb_start_s']) < 30.-1e-6 for x in chosen):
            continue
        chosen.append(r)
        if len(chosen) == limit:
            break
    return chosen


def make_plan(cfg, project_root):
    """Metadata only. No historical embargo or other comparison data is consulted."""
    root = Path(project_root)
    refs = read_json(root/cfg['source_run']/'cohort.json')['refs']
    records = []
    for ds in cfg['measured']['datasets']:
        candidates = [r for r in refs if r['dataset'] == ds and
                      r['record'] == cfg['measured']['records'][ds] and r['site'] == cfg['measured']['site']]
        for subject in sorted({r['subject'] for r in candidates}):
            chosen = nonoverlap([r for r in candidates if r['subject'] == subject],
                                cfg['measured']['windows_per_record'])
            records.append(dict(key=chosen[0]['key'], dataset=ds, subject=subject,
                record=chosen[0]['record'], split=phase(subject), windows=chosen,
                join_key='|'.join((ds, subject, chosen[0]['record']))))
    if not records:
        raise ValueError('no registered public records')
    for ds in cfg['measured']['datasets']:
        train = {r['subject'] for r in records if r['dataset'] == ds and r['split'] == 'train'}
        test = {r['subject'] for r in records if r['dataset'] == ds and r['split'] == 'evaluate'}
        if not train or not test or train & test:
            raise ValueError('subject partition is invalid')
    return dict(records=records, expected_records=len(records),
        expected_windows=sum(len(r['windows']) for r in records),
        split=cfg['measured']['split'], use=cfg['measured']['use'],
        metadata_source=str(root/cfg['source_run']/'cohort.json'))


def software_probe(cfg):
    x = np.arange(40.).reshape(20, 2)
    l = lagged(x, [2, -2])
    assert np.isnan(l[:2, :2]).all() and np.isnan(l[-2:, 2:]).all()
    assert np.array_equal(l[2:, :2], x[:-2])
    candidates = np.stack((np.zeros((120, 2)), np.tile([1., -1.], (120, 1))))
    s = coordinate_stability(candidates, np.ones(2))
    assert s['rms_range_SD'][0] == 0 and s['rms_range_SD'][1] == 1
    y = np.array([[0., 1.]])
    pred = np.array([y, -y])
    assert candidate_inference(y, pred, .1)['best'] == 0
    z = candidate_inference(y, pred, 10., anchors=np.array([1.]),
                            anchor_predictions=np.array([[0.], [1.]]), anchor_sd=.1)
    assert z['best'] == 1
    return dict(status='passed', measured_reads=0, checks=['nonwrapping_lags',
        'amplitude_preserving_Hb_coordinates', 'candidate_inference', 'independent_anchor_score'],
        experiment_id=cfg['experiment_id'])


def prepare_worker(payload):
    cfg, project_root, out, spec = payload
    path = Path(out)/'prepared'/f"{spec['key']}.json"
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_eeg_record, load_native_fnirs_record, ChannelGeometryIndex
    from src.data.physiology_measurement_adapter import measurement_unit_conversion
    root = Path(project_root)
    if spec['dataset'] not in cfg['measured']['datasets'] or spec['record'] != cfg['measured']['records'][spec['dataset']]:
        raise ValueError('record outside the public experiment scope')
    if phase(spec['subject']) != spec['split']:
        raise ValueError('split identity changed before read')
    index = CleanPhysiologyCacheIndex(root/cfg['cache_root'])
    record = next(r for r in index.records if r.join_key == spec['join_key'])
    eeg = load_native_eeg_record(root, record)
    native = load_native_fnirs_record(root, record)
    if eeg.sample_rate_hz != cfg['tensor']['eeg_rate_hz']:
        raise ValueError('native EEG rate differs from declared tensor contract')
    lookup = {n.upper(): i for i, n in enumerate(eeg.channel_names)}
    channels = spec['windows'][0]['eeg_channels']
    selected = [lookup[n.upper()] for n in channels]
    auxiliary = [n.upper() for n in eeg.auxiliary_channel_names]
    # Canonical reader supplies recorded EOG; never synthesize a reference from EEG.
    heog = next((i for i, n in enumerate(auxiliary) if 'HEOG' in n), None)
    veog = next((i for i, n in enumerate(auxiliary) if 'VEOG' in n), None)
    eog_available = eeg.auxiliary_values is not None and heog is not None and veog is not None
    unit = measurement_unit_conversion(eeg.native_unit, quantity='electric_potential',
        evidence=eeg.unit_evidence, group=spec['dataset'])
    op = native_feature_operators()
    geometry = ChannelGeometryIndex(root/cfg['cache_root']).for_channels(record=record,
        modality='fnirs', channel_names=native['channel_names'][::2])
    names = [n.rsplit('_', 1)[0] for n in native['channel_names'][::2]]
    target = names.index(spec['windows'][0]['hb_channel'])
    sos = butter(4, cfg['events']['voltage_band_hz'], btype='bandpass', fs=eeg.sample_rate_hz, output='sos')
    saved = {k: [] for k in ('eeg_raw', 'eeg_voltage', 'eog_voltage', 'hb', 'od_jump')}
    windows, failures = [], []
    for ref in spec['windows']:
        start = round(ref['eeg_start_s']*eeg.sample_rate_hz)
        raw = eeg.values[start:start+6000, selected]*unit['factor']
        times = ref['hb_start_s']+np.arange(300)/10.
        if (raw.shape != (6000, 6) or not np.isfinite(raw).all() or
                times[0] < native['time_s'][0]-1e-8 or times[-1] > native['time_s'][-1]+1e-8):
            failures.append(dict(id=ref['id'], reason='native_time_or_voltage_support'))
            continue
        flat = native['values'].reshape(len(native['time_s']), -1)
        hb = np.column_stack([np.interp(times, native['time_s'], flat[:, i]) for i in range(flat.shape[1])])
        # Missing native support invalidates that channel pair, not unrelated
        # EEG or other Hb channels. The dense temporal operator propagates a
        # missing sample only within its own channel; no imputation is made.
        hb = (op['fnirs']@hb).reshape(120, len(names), 2)
        voltage = resample_poly(sosfiltfilt(sos, raw, axis=0), 1, 4, axis=0)
        eog = np.full((1500, 2), np.nan)
        if eog_available:
            aux = eeg.auxiliary_values[start:start+6000, [heog, veog]]*unit['factor']
            if aux.shape == (6000, 2) and np.isfinite(aux).all():
                eog = resample_poly(sosfiltfilt(sos, aux, axis=0), 1, 4, axis=0)
        jump = np.full((120, len(names)), np.nan)
        optical = native['optical_intensity']
        if optical is not None:
            # log differences cancel the unknown intensity baseline. Positive support only.
            xx = optical.reshape(len(native['time_s']), -1)
            intensity = np.column_stack([np.interp(times, native['time_s'], xx[:, i]) for i in range(xx.shape[1])])
            supported = np.where(np.isfinite(intensity) & (intensity > 0), intensity, np.nan)
            od = -np.log(supported)
            od_diff = np.diff(od, axis=0, prepend=od[:1])
            magnitude = np.sqrt(np.mean(od_diff.reshape(300, len(names), 2)**2, axis=2))
            jump = np.column_stack([np.interp(np.arange(120)/4., np.arange(300)/10., magnitude[:, i])
                                    for i in range(len(names))])
        for k, v in dict(eeg_raw=raw, eeg_voltage=voltage, eog_voltage=eog, hb=hb, od_jump=jump).items():
            saved[k].append(v)
        windows.append(ref)
    if not windows:
        raise ValueError('no valid native windows')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), **{k: np.array(v) for k, v in saved.items()})
    result = dict(status='completed', spec=spec, windows=windows, failures=failures,
        planned_windows=len(spec['windows']), available_windows=len(windows),
        eeg_channels=channels, eog_channels=eeg.auxiliary_channel_names,
        eog_available=eog_available, eeg_unit=unit, hb_provenance=native['provenance'],
        optical_available=native['optical_intensity'] is not None,
        eeg_source=str(eeg.source_path), fnirs_names=names, geometry=geometry,
        target_pair=target, seconds=time.monotonic()-started,
        valid_Hb_pairs_per_window=[np.isfinite(v).all(axis=(0, 2)).sum() for v in saved['hb']],
        valid_target_Hb_windows=sum(np.isfinite(v[:, target]).all() for v in saved['hb']),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        processing='native_unified_reader; window_local_zero_phase_voltage_filter; original_Hb_none_operator; first_5s_reference',
        interpretation='offline_public_development; no causal_forecast_or_source_truth')
    write_json(path, result)
    return result


def stability_worker(payload):
    cfg, project_root, out, ref = payload
    path = Path(out)/'stability'/f"{ref['id']}.json"
    if path.exists():
        return read_json(path)
    old = Path(project_root)/cfg['attribution_run']
    robust = Path(project_root)/cfg['robustness_run']
    audit = read_json(old/'audit'/ref['id']/'result.json')
    scale = np.asarray(audit['diagnostic_scale'])
    with np.load(old/'audit'/ref['id']/'result.npz', allow_pickle=False) as f:
        alternatives = {k: f[k] for k in f.files}
    with np.load(robust/'measured'/ref['id']/'result.npz', allow_pickle=False) as f:
        original = {k: f[k] for k in f.files}
    with np.load(old/'measured'/ref['id']/'result.npz', allow_pickle=False) as f:
        measured = {k: f[k] for k in f.files}
    rows, saved = [], {}
    threshold = cfg['stability']['stable_rms_range_training_SD']
    def append_set(kind, arm, field, values, expected, **metadata):
        if not values:
            for coordinate in ('HbT', 'HbX'):
                rows.append(dict(kind=kind, arm=arm, field=field, coordinate=coordinate, available=False,
                    expected_candidates=expected, candidate_count=0, sufficient_candidates=False,
                    rms_range_SD=np.nan, stable_fraction=0., stable_coordinate=False,
                    peak_time_range_s=np.nan, minimum_reference_shape_cosine=np.nan, **metadata))
            return
        stats = coordinate_stability(np.array(values), scale, threshold=threshold)
        key = f'{kind}__{arm}__{field}'
        for name in ('lower', 'upper', 'pointwise_stable'):
            saved[f'{key}__{name}'] = stats[name]
        saved[f'{key}__candidates'] = np.array(values)
        for j, coordinate in enumerate(('HbT', 'HbX')):
            cosines = stats['shape_cosines'][:, j]
            rows.append(dict(kind=kind, arm=arm, field=field, coordinate=coordinate,
                available=True, expected_candidates=expected, candidate_count=len(values),
                sufficient_candidates=stats['sufficient_candidates'],
                rms_range_SD=stats['rms_range_SD'][j], stable_fraction=stats['stable_fraction'][j],
                amplitude_min_SD=stats['amplitude_min_SD'][j], amplitude_max_SD=stats['amplitude_max_SD'][j],
                peak_time_range_s=stats['peak_time_range_s'][j],
                minimum_reference_shape_cosine=float(np.min(cosines[np.isfinite(cosines)])) if np.isfinite(cosines).any() else np.nan,
                stable_coordinate=bool(stats['stable_coordinate'][j]), **metadata))
    for arm in ('M-observation', 'M-combined'):
        eligible = [r for r in audit['sensitivity'] if r['arm'] == arm and r['converged']]
        expected = sum(r['arm'] == arm for r in audit['sensitivity'])+1
        for field in ('physical_prediction', 'observation_component'):
            values = [original[f'full__{arm}__{field}'][:, 1:]]
            values += [alternatives[f"{arm}__{r['variant']}__{field}"][:, 1:] for r in eligible]
            append_set('sensitivity', arm, field, values, expected,
                       comparison='different_measurement_prior_or_loading_settings_not_near_equivalent_solutions')
    profiles = old/'profiles'/ref['id']/'result.json'
    if profiles.exists():
        profile = read_json(profiles)
        with np.load(profiles.with_suffix('.npz'), allow_pickle=False) as f:
            for arm in ('D-fixed', 'D-trained', 'D-exchange'):
                eligible = [r for r in profile['rows'] if r['arm'] == arm and r['near_equivalent']]
                for field in ('prediction', 'physical_prediction', 'observation_component'):
                    # Original optimum plus nonzero admitted displacements; zero is not a second explanation.
                    values = [measured[f'full__{arm}__{field}'][:, 1:]]
                    for r in eligible:
                        if r['offset_SD'] != 0:
                            values.append(f"{arm}__{r['offset_SD']}__{field}")
                            values[-1] = f[values[-1]][:, 1:]
                    append_set('near_equivalent_profile', arm, field, values, 5,
                               admitted_nonzero=sum(r['offset_SD'] != 0 for r in eligible),
                               comparison='same_objective_5percent_sampled_profile_not_confidence_interval')
    strengthened = Path(project_root)/cfg['strengthening_run']
    allowed_arrays = (Path(project_root)/cfg['strengthening_array_run']).resolve()
    def strengthened_arrays(meta, folder):
        filename = Path(meta['array_path']).resolve()
        if filename != allowed_arrays/folder/f"{ref['id']}.npz":
            raise ValueError('unexpected strengthened evidence array identity')
        return np.load(filename, allow_pickle=False)
    response = read_json(strengthened/'measured'/f"{ref['id']}.json")
    eligible = {r['arm'] for r in response['rows'] if r['mode'] == 'full' and r['pairing'] == 'real' and r['converged']}
    with strengthened_arrays(response, 'measured') as f:
        for field in ('prediction', 'physical_prediction', 'observation_component'):
            values = [f[f'0__full__{arm}__{field}'][:, 1:] for arm in ('BC', 'HselectedC') if arm in eligible]
            if len(values) == 2 and np.allclose(values[0], values[1], rtol=0, atol=1e-12):
                values = values[:1]
            append_set('response_choice', 'BC_vs_HselectedC', field, values, 2,
                comparison='training_selected_response_not_same_objective_profile')
        profile_path = strengthened/'profiles'/f"{ref['id']}.json"
        if profile_path.exists():
            profile = read_json(profile_path)
            with strengthened_arrays(profile, 'profiles') as prof:
                for arm in ('BC', 'HselectedC'):
                    if arm not in eligible:
                        continue
                    for group in ('initial', 'response'):
                        admitted = [r for r in profile['rows'] if r['arm'] == arm and r['mode'] == 'full'
                                    and r['group'] == group and r.get('near_equivalent') and r['offset_sd'] != 0]
                        for field in ('prediction', 'physical_prediction', 'observation_component'):
                            values = [f[f'0__full__{arm}__{field}'][:, 1:]]
                            values += [prof[f"{arm}__full__{group}__{r['offset_sd']}__{field}"][:, 1:] for r in admitted]
                            append_set('near_equivalent_'+group, arm, field, values, 5,
                                comparison='tangent_guided_component_displacement_physical_refit_fixed_response_parameters')
    saved['observed_coordinates'] = hb_diagnostic_coordinates(measured['target'][:, 1:])
    saved['coordinate_scale'] = scale
    if 'hidden_channel__spatial_ridge__prediction' in measured:
        saved['independent_spatial_coordinates'] = hb_diagnostic_coordinates(measured['hidden_channel__spatial_ridge__prediction'][:, 1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), **saved)
    result = dict(status='completed', ref=ref, rows=rows, source=str(old/'audit'/ref['id']),
        scale_source='retained_training_HbT_HbX_SD',
        teacher_fields='coordinate_lower_upper; pointwise_stable; sufficient_candidates; candidate_count; source_unresolved')
    write_json(path, result)
    return result


def task_worker(kind, payload):
    started = time.monotonic()
    try:
        result = dict(prepare_worker=prepare_worker, stability_worker=stability_worker,
                      mechanism_worker=mechanism_worker)[kind](payload)
        return result
    except Exception:
        cfg, project_root, out, spec = payload
        identity = spec.get('key', spec.get('id', str(spec.get('repeat'))))
        failure = dict(status='failed', identity=identity, stage=kind,
            seconds=time.monotonic()-started, traceback=traceback.format_exc())
        write_json(Path(out)/'failures'/f'{kind}__{identity}.json', failure)
        return failure


def run_tasks(kind, payloads, out, workers):
    result = []
    started = time.monotonic()
    # Executor has at most 2*workers submitted tasks, bounding arrays and memory.
    iterator = iter(payloads)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        def fill():
            while len(futures) < 2*workers:
                try:
                    payload = next(iterator)
                except StopIteration:
                    break
                futures[pool.submit(task_worker, kind, payload)] = payload[-1]
        fill()
        while futures:
            future = next(as_completed(futures))
            futures.pop(future)
            item = future.result()
            result.append(item)
            elapsed = time.monotonic()-started
            write_json(Path(out)/f'{kind}_progress.json', dict(completed=len(result),
                planned=len(payloads), failed=sum(v['status'] != 'completed' for v in result),
                elapsed_s=elapsed, eta_s=elapsed/len(result)*(len(payloads)-len(result))))
            print(f'{kind}: {len(result)}/{len(payloads)} {item["status"]}', flush=True)
            fill()
    return result


def load_prepared(out, plan, ds):
    records, windows = [], []
    for spec in plan['records']:
        if spec['dataset'] != ds:
            continue
        path = Path(out)/'prepared'/f"{spec['key']}.json"
        if not path.exists():
            continue
        meta = read_json(path)
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        records.append(meta)
        for i, ref in enumerate(meta['windows']):
            windows.append(dict(ref=ref, split=spec['split'], subject=spec['subject'], meta=meta,
                                **{k: v[i] for k, v in arrays.items()}))
    return records, windows


def score_mask(rate, interval):
    t = np.arange(round(30*rate))/rate
    return (t >= interval[0]) & (t < interval[1])


def nrmse(pred, target, sd, mask):
    p, y = np.asarray(pred), np.asarray(target)
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        return np.nan
    return float(np.sqrt(np.mean(((p[mask]-y[mask])/sd)**2)))


def condition_templates(training, field):
    result = {}
    for condition in sorted({w['ref']['condition'] for w in training}):
        values = np.array([w[field] for w in training if w['ref']['condition'] == condition])
        count = np.isfinite(values).sum(axis=0)
        result[condition] = np.divide(np.nansum(values, axis=0), count,
            out=np.full(values.shape[1:], np.nan), where=count > 0)
    return result


def wrong_window(training, target):
    choices = sorted([w for w in training if w['ref']['condition'] == target['ref']['condition'] and
                      w['subject'] != target['subject']], key=lambda w: w['ref']['id'])
    if not choices:
        return None
    return choices[int(target['ref']['window']) % len(choices)]


def summarize_contrasts(frame, groups, contrasts, *, value='nrmse', unit='subject', seed=20261009):
    """Within-window pairing, then equal-subject averaging and block inference."""
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(groups, keys))
        for label, control, candidate in contrasts:
            a = part[part['arm'] == control][['id', unit, value]]
            b = part[part['arm'] == candidate][['id', unit, value]]
            pair = a.merge(b, on=['id', unit], suffixes=('_control', '_candidate'), validate='one_to_one')
            pair['gain'] = pair[value+'_control']-pair[value+'_candidate']
            paired_finite = pair.dropna(subset=['gain'])
            units = paired_finite.groupby(unit)['gain'].mean()
            info = block_interval(units, seed=seed)
            rows.append(dict(**metadata, contrast=label, control=control, candidate=candidate,
                planned_pairs=len(pair), available_pairs=len(paired_finite),
                planned_units=pair[unit].nunique(), **info, p=sign_flip_p(units, seed=seed)))
    result = pd.DataFrame(rows)
    if not result.empty:
        result['holm_p'] = holm(result['p'])
    return result


def event_analysis(cfg, out, plan):
    rows, event_rows, chain_rows, curves = [], [], [], {}
    frozen = {}
    e = cfg['events']
    for ds in cfg['measured']['datasets']:
        records, windows = load_prepared(out, plan, ds)
        usable = [w for w in windows if np.isfinite(w['eog_voltage']).all()]
        training = [w for w in usable if w['split'] == 'train']
        evaluation = [w for w in usable if w['split'] == 'evaluate']
        if not training or not evaluation:
            frozen[ds] = dict(status='unavailable', reason='no_train_or_evaluation_EOG')
            continue
        scale = np.median([np.std(np.column_stack((w['eog_voltage'][:, 1],
                np.gradient(w['eog_voltage'][:, 0])*50)), axis=0) for w in training], axis=0)
        scale = np.maximum(scale, 1e-8)
        shifts = [round(v*50) for v in e['voltage_lags_s']]
        vm = score_mask(50, e['score_interval_s'])
        hm = score_mask(4, e['score_interval_s'])
        # Voltage mapping sees EOG only; all six EEG targets are held out on evaluation subjects.
        x = np.concatenate([lagged(w['eog_voltage'], shifts)[vm] for w in training])
        y = np.concatenate([w['eeg_voltage'][vm] for w in training])
        voltage_model = fit_standardized_ridge(x, y, e['ridge_weight'])
        voltage_sd = np.maximum(y.std(axis=0), 1e-8)
        for w in usable:
            impulse, events = event_impulses(w['eog_voltage'], scale, rate=50,
                height=e['event_height_training_SD'], refractory=e['event_refractory_s'])
            # Exact bin membership, avoiding peak duplication by interpolation.
            bins = np.minimum((np.arange(len(impulse))/50*4).astype(int), 119)
            imp4 = np.zeros((120, 2))
            np.add.at(imp4, bins, impulse)
            w['events'] = events
            w['event_features'] = lagged(imp4, [round(v*4) for v in e['instant_lags_s']+e['delayed_lags_s']])
            w['Hb_coordinates'] = hb_diagnostic_coordinates(w['hb'][:, w['meta']['target_pair']])
            w['OD_jump'] = w['od_jump'][:, w['meta']['target_pair'], None]
            for event in events:
                event_rows.append(dict(dataset=ds, subject=w['subject'], id=w['ref']['id'],
                    split=w['split'], condition=w['ref']['condition'], time_s=event['sample']/50,
                    kind=event['kind'], signed_height=event['signed_height']))
        models = {}
        for field in ('Hb_coordinates', 'OD_jump'):
            supported = [w for w in training if np.isfinite(w[field]).all()]
            if len(supported) < 2:
                continue
            templates = condition_templates(supported, field)
            xx = np.concatenate([w['event_features'][hm] for w in supported])
            yy = np.concatenate([(w[field]-templates[w['ref']['condition']])[hm] for w in supported])
            model = fit_standardized_ridge(xx, yy, e['ridge_weight'])
            instant_model = fit_standardized_ridge(xx[:, :8], yy, e['ridge_weight'])
            sd = np.maximum(np.concatenate([w[field][hm] for w in supported]).std(axis=0), 1e-12)
            models[field] = dict(model=model, instant_model=instant_model, templates=templates, sd=sd,
                training_windows=len(supported), training_subjects=sorted({w['subject'] for w in supported}))
        frozen[ds] = dict(status='completed', training_subjects=sorted({w['subject'] for w in training}),
            evaluation_subjects=sorted({w['subject'] for w in evaluation}), training_windows=len(training),
            evaluation_windows=len(evaluation), EOG_event_training_scale=scale, voltage_model=voltage_model,
            voltage_sd=voltage_sd, endpoint_models=models)
        for w in evaluation:
            wrong = wrong_window(training, w)
            if wrong is None:
                continue
            base = dict(dataset=ds, subject=w['subject'], id=w['ref']['id'], condition=w['ref']['condition'])
            predictions = {}
            for arm, aux in [('EOG_real', w['eog_voltage']), ('EOG_wrong_subject', wrong['eog_voltage']),
                             ('EOG_time_shift', lagged(w['eog_voltage'], [-round(e['null_shift_s']*50)]))]:
                pred = predict_standardized_ridge(voltage_model, lagged(aux, shifts))
                predictions[arm] = pred
                rows.append(dict(**base, endpoint='EEG_voltage', arm=arm,
                    nrmse=nrmse(pred, w['eeg_voltage'], voltage_sd, vm)))
            rows.append(dict(**base, endpoint='EEG_voltage', arm='training_mean',
                nrmse=nrmse(np.broadcast_to(voltage_model['target_mean'], y[:1500].shape), w['eeg_voltage'], voltage_sd, vm)))
            for field, frozen_model in models.items():
                template = frozen_model['templates'].get(w['ref']['condition'], np.full_like(w[field], np.nan))
                for arm, features in [('EOG_real', w['event_features']), ('EOG_wrong_subject', wrong['event_features']),
                    ('EOG_time_shift', lagged(w['event_features'], [-round(e['null_shift_s']*4)]))]:
                    pred = template+predict_standardized_ridge(frozen_model['model'], features)
                    for j in range(w[field].shape[1]):
                        endpoint = ('HbT', 'HbX')[j] if field == 'Hb_coordinates' else 'OD_jump'
                        rows.append(dict(**base, endpoint=endpoint, arm=arm,
                            nrmse=nrmse(pred[:, j], w[field][:, j], frozen_model['sd'][j], hm)))
                for j in range(w[field].shape[1]):
                    endpoint = ('HbT', 'HbX')[j] if field == 'Hb_coordinates' else 'OD_jump'
                    rows.append(dict(**base, endpoint=endpoint, arm='training_mean',
                        nrmse=nrmse(template[:, j], w[field][:, j], frozen_model['sd'][j], hm)))
                    pred = template+predict_standardized_ridge(frozen_model['instant_model'], w['event_features'][:, :8])
                    rows.append(dict(**base, endpoint=endpoint, arm='EOG_instant',
                        nrmse=nrmse(pred[:, j], w[field][:, j], frozen_model['sd'][j], hm)))
            # Frozen voltage subtraction followed by the existing exact five-band feature chain.
            # Guard-window interpolation only for this descriptive propagation endpoint.
            prediction = predictions['EOG_real'].copy()
            for j in range(6):
                good = np.isfinite(prediction[:, j])
                prediction[:, j] = np.interp(np.arange(1500), np.flatnonzero(good), prediction[good, j])
            electrical = resample_poly(prediction, 4, 1, axis=0)
            raw = w['eeg_raw']
            power_raw = eeg_band_power(raw, bands=cfg['tensor']['eeg_bands_hz'], sample_rate=200, target_rate=4)
            power_eog = eeg_band_power(electrical, bands=cfg['tensor']['eeg_bands_hz'], sample_rate=200, target_rate=4)
            power_remainder = eeg_band_power(raw-electrical, bands=cfg['tensor']['eeg_bands_hz'], sample_rate=200, target_rate=4)
            for j, band in enumerate(cfg['tensor']['eeg_bands_hz']):
                chain_rows.append(dict(**base, band=f'{band[0]}-{band[1]}Hz',
                    logpower_change=float(np.mean(np.log(np.maximum(power_remainder[hm, :, j], 1e-12))-
                        np.log(np.maximum(power_raw[hm, :, j], 1e-12)))),
                    power_cross_term_relative_RMS=float(np.sqrt(np.mean((power_raw[hm, :, j]-power_eog[hm, :, j]-
                        power_remainder[hm, :, j])**2))/max(np.sqrt(np.mean(power_raw[hm, :, j]**2)), 1e-12))))
            # Per-subject event means; no event is treated as an independent subject.
            event_times = [x['sample']/50 for x in w['events'] if x['kind'] == 0 and 2 <= x['sample']/50 < 20]
            ek = f"{ds}__{w['subject']}"
            for t in event_times:
                idx = np.round((t+np.arange(-8, 41)/4)*4).astype(int)
                if idx.min() < 0 or idx.max() >= 120:
                    continue
                curves.setdefault(ek, []).append(np.column_stack((
                    w['Hb_coordinates'][idx], w['OD_jump'][idx],
                    np.interp(idx/4, np.arange(1500)/50, w['eog_voltage'][:, 1]))))
    pd.DataFrame(rows).to_csv(Path(out)/'event_metrics.csv', index=False)
    pd.DataFrame(event_rows).to_csv(Path(out)/'event_inventory.csv', index=False)
    pd.DataFrame(chain_rows).to_csv(Path(out)/'power_chain.csv', index=False)
    write_json(Path(out)/'event_models.json', frozen)
    means = {}
    for k, v in curves.items():
        values = np.array(v)
        counts = np.isfinite(values).sum(axis=0)
        means[k] = np.divide(np.nansum(values, axis=0), counts,
            out=np.full(values.shape[1:], np.nan), where=counts > 0)
    np.savez_compressed(Path(out)/'event_triggered.npz', **means, time_s=np.arange(-8, 41)/4)
    frame = pd.DataFrame(rows)
    comparisons = summarize_contrasts(frame, ['dataset', 'endpoint'], [
        ('real_vs_template', 'training_mean', 'EOG_real'),
        ('real_vs_wrong', 'EOG_wrong_subject', 'EOG_real'),
        ('real_vs_shift', 'EOG_time_shift', 'EOG_real'),
        ('delayed_increment', 'EOG_instant', 'EOG_real')])
    comparisons = comparisons[~((comparisons.endpoint == 'EEG_voltage') & (comparisons.contrast == 'delayed_increment'))].copy()
    # Electrical association is the primary family; Hb and OD are secondary families.
    for primary in (True, False):
        mask = comparisons.endpoint.eq('EEG_voltage') == primary
        comparisons.loc[mask, 'holm_p'] = holm(comparisons.loc[mask, 'p'])
        comparisons.loc[mask, 'family'] = 'primary_voltage' if primary else 'secondary_Hb_OD'
    comparisons.to_csv(Path(out)/'event_comparisons.csv', index=False)
    return dict(status='completed', rows=len(rows), event_count=len(event_rows),
        datasets={k: {f: v.get(f) for f in ('status', 'training_windows', 'evaluation_windows',
                   'training_subjects', 'evaluation_subjects')} for k, v in frozen.items()})


def hardware_analysis(cfg, out, plan):
    ds = cfg['hardware']['dataset']
    records, windows = load_prepared(out, plan, ds)
    training = [w for w in windows if w['split'] == 'train']
    evaluation = [w for w in windows if w['split'] == 'evaluate']
    geometry = records[0]['geometry']
    names = records[0]['fnirs_names']
    # Require the exact same hardware IDs and channel names across participants.
    for r in records:
        if r['fnirs_names'] != names or [(v['source_index'], v['detector_index']) for v in r['geometry']] != [
                (v['source_index'], v['detector_index']) for v in geometry]:
            raise ValueError('hardware topology is not invariant across records')
    hm = score_mask(4, cfg['hardware']['score_interval_s'])
    templates = condition_templates(training, 'hb')
    rows, topology, frozen, examples = [], [], {}, {}
    for target in range(len(names)):
        selection = hardware_donors(geometry, target)
        topology.append(dict(target=names[target], target_index=target, **selection))
        if not selection['available']:
            continue
        pair_indices = [target]+list(selection['indices'].values())
        supported = [w for w in training if np.isfinite(w['hb'][:, pair_indices]).all()]
        if len(supported) < 2:
            topology[-1].update(available=False, reason='insufficient_common_training_support')
            continue
        y = np.concatenate([(w['hb']-templates[w['ref']['condition']])[hm, target] for w in supported])
        sd = np.maximum(np.concatenate([w['hb'][hm, target] for w in supported]).std(axis=0), 1e-12)
        models = {}
        for arm, donor in selection['indices'].items():
            x = np.concatenate([w['hb'][hm, donor] for w in supported])
            models[arm] = fit_standardized_ridge(x, y, cfg['hardware']['ridge_weight'])
        frozen[names[target]] = dict(donors=selection, models=models, sd=sd,
            training_windows=len(supported), training_subjects=sorted({w['subject'] for w in supported}))
        for w in evaluation:
            wrong = wrong_window(training, w)
            base = dict(dataset=ds, subject=w['subject'], id=w['ref']['id']+'__'+names[target],
                window_id=w['ref']['id'], target=names[target], condition=w['ref']['condition'])
            target_values = w['hb'][:, target]
            template = templates[w['ref']['condition']][:, target]
            rows.append(dict(**base, arm='training_template', pairing='real',
                nrmse=nrmse(template, target_values, sd, hm)))
            for arm, donor in selection['indices'].items():
                for pairing, features in [('real', w['hb'][:, donor]),
                    ('time_shift', lagged(w['hb'][:, donor], [-round(cfg['hardware']['null_shift_s']*4)])),
                    ('wrong_subject', wrong['hb'][:, donor] if wrong else np.full((120, 2), np.nan))]:
                    pred = template+predict_standardized_ridge(models[arm], features)
                    label = arm if pairing == 'real' else arm+'__'+pairing
                    target_jump = w['od_jump'][hm, target]
                    donor_jump = w['od_jump'][hm, donor]
                    finite = np.isfinite(target_jump) & np.isfinite(donor_jump)
                    jump_correlation = (float(np.corrcoef(target_jump[finite], donor_jump[finite])[0, 1])
                        if finite.sum() > 10 and np.std(target_jump[finite])*np.std(donor_jump[finite]) > 1e-20 else np.nan)
                    rows.append(dict(**base, arm=label, pairing=pairing, donor=names[donor],
                        distance=selection['distances'][arm], nrmse=nrmse(pred, target_values, sd, hm),
                        synchronous_OD_jump_correlation=jump_correlation if pairing == 'real' else np.nan,
                        target_OD_jump_RMS=float(np.sqrt(np.nanmean(target_jump**2))) if np.isfinite(target_jump).any() else np.nan))
                    if target == 0 and w is evaluation[0]:
                        examples[label] = pred
            if target == 0 and w is evaluation[0]:
                examples.update(target=target_values, template=template, sd=sd)
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out)/'hardware_metrics.csv', index=False)
    contrasts = [('shared_vs_disjoint', 'disjoint_distance_matched', 'shared_optode'),
                 ('near_vs_far_disjoint', 'disjoint_far', 'disjoint_distance_matched')]
    for arm in cfg['hardware']['arms']:
        contrasts += [(arm+'_vs_template', 'training_template', arm),
                      (arm+'_vs_wrong', arm+'__wrong_subject', arm),
                      (arm+'_vs_shift', arm+'__time_shift', arm)]
    summary = summarize_contrasts(frame, ['dataset'], contrasts)
    summary.to_csv(Path(out)/'hardware_comparisons.csv', index=False)
    write_json(Path(out)/'hardware_topology.json', topology)
    write_json(Path(out)/'hardware_models.json', dict(targets=frozen, templates=templates,
        training_subjects=sorted({w['subject'] for w in training}),
        evaluation_subjects=sorted({w['subject'] for w in evaluation})))
    np.savez_compressed(Path(out)/'hardware_example.npz', **examples)
    return dict(status='completed', target_count=len(topology),
        available_targets=sum(r['available'] for r in topology), rows=len(rows),
        planned_target_windows=len(geometry)*len(evaluation),
        available_target_windows=int(frame.loc[frame.arm == 'shared_optode', 'nrmse'].notna().sum()),
        planned_subjects=len([r for r in plan['records'] if r['dataset'] == ds and r['split'] == 'evaluate']),
        available_subjects=len({w['subject'] for w in evaluation}),
        other_dataset_status='Simultaneous_montage_has_no_verified_source_detector_ids')


def mechanism_library(cfg, repeat, split):
    """Known-driver/initial-state oracle upper bound using the actual nonlinear SSM."""
    from src.inference.t3a_balloon_robust_ssm import BalloonParameters, BalloonFreeParameters
    from src.inference.semantic_response_ssm import fixed_driver_response
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 901 if split == 'train' else 902, repeat]))
    driver = gaussian_filter1d(rng.normal(size=144), 5.)
    driver += .25*gaussian_filter1d(rng.normal(size=144), 1.5)
    driver *= .025/driver.std()
    initial = np.r_[0., np.ones(4)]+rng.normal(size=5)*.005
    parameters = BalloonParameters(free=BalloonFreeParameters(kappa=.64, tau=2.))
    baseline_long = fixed_driver_response(driver, parameters, .25, initial_state=initial, substeps=8)['prediction']
    # Same temporal operator and physical reference for all mechanisms.
    op = native_feature_operators()
    hbop = op['fnirs']@op['native_interpolation']
    base = hbop@baseline_long[12:132]
    entries = []
    common = gaussian_filter1d(rng.normal(size=120), 8.)
    common = hbop@common
    common /= max(np.std(common), 1e-12)
    for mechanism in cfg['mechanisms']['mechanisms']:
        values = dict(tau=np.geomspace(.7, 5.7, 9), Hb_gain=np.linspace(.5, 1.5, 9),
                      extra_HbT=np.linspace(-.04, .04, 9), lag=np.linspace(-3., 3., 9))[mechanism]
        for value in values:
            physical = base.copy()
            predicted = base.copy()
            gain, lag = 0., 0.
            extra = np.zeros_like(base)
            if mechanism == 'tau':
                p = replace(parameters, free=replace(parameters.free, tau=float(value)))
                physical = hbop@fixed_driver_response(driver, p, .25, initial_state=initial, substeps=8)['prediction'][12:132]
                predicted = physical.copy()
            elif mechanism == 'Hb_gain':
                gain = float(value)-1.
                predicted = (1.+gain)*base
            elif mechanism == 'extra_HbT':
                extra = float(value)*common[:, None]*np.array([.65, .35])
                predicted += extra
            elif mechanism == 'lag':
                lag = float(value)
                t = 3.+np.arange(120)/4+lag
                shifted = np.column_stack([np.interp(t, np.arange(144)/4, baseline_long[:, j]) for j in range(2)])
                predicted = hbop@shifted
            # Independent synthetic calibration quantities, never a tau/class label.
            anchor = np.r_[gain, lag, [np.mean(extra[a:b].sum(axis=1)) for a, b in ((0, 30), (30, 60), (60, 90), (90, 120))]]
            entries.append(dict(mechanism=mechanism, parameter=float(value), prediction=predicted,
                physical=physical, component=predicted-physical, anchor=anchor))
    return dict(entries=entries, baseline=base, driver=driver[12:132], initial=initial)


def calibrate_mechanisms(cfg, out):
    bases, anchors = [], []
    for repeat in range(cfg['mechanisms']['training_identities']):
        lib = mechanism_library(cfg, repeat, 'train')
        bases.append(lib['baseline'])
        anchors.extend(e['anchor'] for e in lib['entries'])
    sd = np.maximum(np.std(bases, axis=(0, 1)), 1e-8)
    anchor_sd = np.maximum(np.std(anchors, axis=0)*cfg['mechanisms']['anchor_noise_fraction'],
                           np.r_[.02, .1, np.full(4, 1e-4)])
    result = dict(status='completed', sd=sd, noise_sd=sd*cfg['mechanisms']['observation_noise_SD'],
        anchor_sd=anchor_sd, training_identities=cfg['mechanisms']['training_identities'],
        no_evaluation_tuning=True, interpretation='synthetic_known_driver_and_initial_state_oracle')
    write_json(Path(out)/'mechanism_calibration.json', result)
    return result


def mechanism_worker(payload):
    cfg, project_root, out, spec = payload
    repeat = spec['repeat']
    path = Path(out)/'mechanisms'/f'r{repeat:03d}.json'
    if path.exists():
        return read_json(path)
    cal = read_json(Path(out)/'mechanism_calibration.json')
    sd, noise, anchor_sd = (np.asarray(cal[k]) for k in ('sd', 'noise_sd', 'anchor_sd'))
    lib = mechanism_library(cfg, repeat, 'evaluate')
    entries = lib['entries']
    predictions = np.array([e['prediction'] for e in entries])
    anchors = np.array([e['anchor'] for e in entries])
    base = lib['baseline']
    change = (predictions-base)/sd
    rms = np.sqrt(np.mean(change**2, axis=(1, 2)))
    # Match generation without observing inference noise or mechanism scores.
    spectra = abs(np.fft.rfft(change, axis=1))**2
    spectra /= np.maximum(spectra.sum(axis=(1, 2), keepdims=True), 1e-12)
    rows, matches, saved = [], [], {}
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 903, repeat]))
    for pair_index, pair in enumerate(cfg['mechanisms']['competing_pairs']):
        first = [i for i, e in enumerate(entries) if e['mechanism'] == pair[0] and rms[i] >= cfg['mechanisms']['min_change_SD']]
        second = [i for i, e in enumerate(entries) if e['mechanism'] == pair[1] and rms[i] >= cfg['mechanisms']['min_change_SD']]
        if not first or not second:
            matches.append(dict(pair='/'.join(pair), available=False, reason='no_nontrivial_candidates'))
            continue
        options = []
        for i in first:
            for j in second:
                distance = float(np.sqrt(np.mean((change[i]-change[j])**2)))
                amp = abs(np.log(rms[i]/rms[j]))
                spectral = float(np.sqrt(np.mean((spectra[i]-spectra[j])**2)))
                target = cfg['mechanisms']['target_change_SD']
                cost = distance+.2*amp+.2*(abs(rms[i]-target)+abs(rms[j]-target))+spectral
                options.append((cost, i, j, distance, amp, spectral))
        cost, a, b, distance, amp, spectral = min(options)
        matches.append(dict(pair='/'.join(pair), available=True, first=a, second=b,
            first_mechanism=pair[0], second_mechanism=pair[1], first_parameter=entries[a]['parameter'],
            second_parameter=entries[b]['parameter'], waveform_distance_SD=distance,
            first_change_SD=rms[a], second_change_SD=rms[b], amplitude_log_ratio=amp,
            spectrum_distance=spectral))
        allowed = [i for i, e in enumerate(entries) if e['mechanism'] in pair]
        for truth_index, wrong_index in ((a, b), (b, a)):
            true = entries[truth_index]
            observed = true['prediction']+rng.normal(size=(120, 2))*noise
            true_anchor = true['anchor']+rng.normal(size=6)*anchor_sd
            wrong_anchor = entries[wrong_index]['anchor']+rng.normal(size=6)*anchor_sd
            for regime in cfg['mechanisms']['anchor_regimes']:
                anchor = true_anchor if regime == 'correct' else wrong_anchor if regime == 'mismatched' else None
                inferred = candidate_inference(observed, predictions[allowed], noise,
                    anchors=anchor, anchor_predictions=anchors[allowed] if anchor is not None else None,
                    anchor_sd=anchor_sd if anchor is not None else None,
                    cutoff=cfg['mechanisms']['likelihood_ratio_cutoff'])
                best = allowed[inferred['best']]
                admitted = [allowed[i] for i in np.flatnonzero(inferred['admissible'])]
                labels = sorted({entries[i]['mechanism'] for i in admitted})
                estimated = entries[best]
                component_error = float(np.sqrt(np.mean(((estimated['component']-true['component'])/sd)**2)))
                physical_error = float(np.sqrt(np.mean(((estimated['physical']-true['physical'])/sd)**2)))
                d = (true['prediction']-base)/sd
                projection = float(np.sum((estimated['component']/sd)*d)/max(np.sum(d*d), 1e-12))
                row = dict(repeat=repeat, subject=f'seed_{repeat:03d}', pair='/'.join(pair),
                    id=f'r{repeat:03d}__p{pair_index}__{true["mechanism"]}', truth=true['mechanism'],
                    arm=regime, predicted=estimated['mechanism'], error=int(estimated['mechanism'] != true['mechanism']),
                    component_error=component_error, physical_error=physical_error,
                    false_component_projection=projection if true['mechanism'] == 'tau' else np.nan,
                    ambiguous=int(len(labels) > 1), set_coverage=int(true['mechanism'] in labels),
                    parameter_set_coverage=int(truth_index in admitted), candidate_set_size=len(admitted),
                    reconstruction_nrmse=float(np.sqrt(np.mean(((estimated['prediction']-observed)/sd)**2))))
                rows.append(row)
                if repeat < 3:
                    prefix = f'p{pair_index}__{true["mechanism"]}__{regime}'
                    saved[prefix+'__prediction'] = estimated['prediction']
                    saved[prefix+'__component'] = estimated['component']
                    saved[prefix+'__physical'] = estimated['physical']
            if repeat < 3:
                prefix = f'p{pair_index}__{true["mechanism"]}'
                saved[prefix+'__observed'] = observed
                saved[prefix+'__true_component'] = true['component']
                saved[prefix+'__truth'] = true['prediction']
    path.parent.mkdir(parents=True, exist_ok=True)
    if saved:
        np.savez_compressed(path.with_suffix('.npz'), **saved, baseline=base, sd=sd)
    result = dict(status='completed', repeat=repeat, rows=rows, matches=matches,
        expected_pairs=len(cfg['mechanisms']['competing_pairs']),
        interpretation='known_driver_and_initial_oracle; independent_synthetic_anchors; not_measured_source_validation')
    write_json(path, result)
    return result


def summarize(cfg, out, plan, project_root=ROOT):
    out = Path(out)
    a_rows = []
    a_files = sorted((out/'stability').glob('*.json'))
    for f in a_files:
        item = read_json(f)
        for row in item['rows']:
            a_rows.append(dict(dataset=item['ref']['dataset'], subject=item['ref']['subject'], id=item['ref']['id'], **row))
    a = pd.DataFrame(a_rows)
    a.to_csv(out/'stability_metrics.csv', index=False)
    a_summary = []
    for key, part in a.groupby(['dataset', 'kind', 'arm', 'field', 'coordinate']):
        for metric in ('rms_range_SD', 'stable_fraction', 'stable_coordinate',
                       'peak_time_range_s', 'minimum_reference_shape_cosine'):
            units = part.groupby('subject')[metric].mean()
            a_summary.append(dict(zip(['dataset', 'kind', 'arm', 'field', 'coordinate'], key),
                metric=metric, windows=len(part), sufficient_windows=int(part.sufficient_candidates.sum()),
                **block_interval(units)))
    pd.DataFrame(a_summary).to_csv(out/'stability_summary.csv', index=False)
    mechanism_results = [read_json(p) for p in sorted((out/'mechanisms').glob('r*.json'))]
    d = pd.DataFrame([row for item in mechanism_results for row in item['rows']])
    d.to_csv(out/'mechanism_metrics.csv', index=False)
    pd.DataFrame([dict(repeat=item['repeat'], **row) for item in mechanism_results for row in item['matches']]).to_csv(out/'mechanism_matching.csv', index=False)
    d_summary = []
    for keys, part in d.groupby(['pair', 'arm']):
        for metric in ('error', 'component_error', 'physical_error', 'ambiguous', 'set_coverage',
                       'parameter_set_coverage', 'reconstruction_nrmse', 'false_component_projection'):
            d_summary.append(dict(pair=keys[0], arm=keys[1], metric=metric,
                **block_interval(part.groupby('subject')[metric].mean())))
    pd.DataFrame(d_summary).to_csv(out/'mechanism_summary.csv', index=False)
    contrasts = []
    for metric in ('error', 'component_error', 'physical_error'):
        comp = summarize_contrasts(d, ['pair'], [('correct_vs_absent', 'absent', 'correct'),
            ('correct_vs_mismatched', 'mismatched', 'correct')], value=metric)
        comp['metric'] = metric
        contrasts.append(comp)
    comparisons = pd.concat(contrasts, ignore_index=True)
    comparisons['holm_p'] = holm(comparisons.p)
    comparisons.to_csv(out/'mechanism_comparisons.csv', index=False)
    prep = [read_json(out/'prepared'/f"{s['key']}.json") for s in plan['records']
            if (out/'prepared'/f"{s['key']}.json").exists()]
    failures = sorted((out/'failures').glob('*.json')) if (out/'failures').exists() else []
    expected_a = len(read_json(Path(project_root)/cfg['attribution_run']/'measured_plan.json')['windows'])
    complete = (len(prep) == plan['expected_records'] and len(a_files) == expected_a and
                len(mechanism_results) == cfg['mechanisms']['evaluation_identities'] and
                all((out/f'{stage}_result.json').exists() for stage in ('event', 'hardware')) and not failures)
    result = dict(experiment_id=cfg['experiment_id'], execution='completed' if complete else 'incomplete',
        updated_at=datetime.now(timezone.utc).isoformat(), scientific_verdict='exploratory_component_specific_evidence',
        A=dict(planned_windows=expected_a, available_windows=len(a_files), metrics='stability_summary.csv'),
        B=read_json(out/'event_result.json'), C=read_json(out/'hardware_result.json'),
        D=dict(planned_identities=cfg['mechanisms']['evaluation_identities'],
               available_identities=len(mechanism_results), cells=len(d), metrics='mechanism_summary.csv'),
        E=dict(status='not_run', reason='no_independent_short_separation_or_peripheral_validation_dataset_supplied'),
        measured=dict(planned_records=plan['expected_records'], available_records=len(prep),
                      planned_windows=plan['expected_windows'], available_windows=sum(x['available_windows'] for x in prep)),
        failures=[str(p.relative_to(out)) for p in failures],
        claims='observation_physical_coordinates_and_tested_associations_only_no_tissue_or_unique_source_labels')
    write_json(out/'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--stage', choices=['check', 'plan', 'pilot', 'prepare', 'stability', 'events', 'hardware', 'mechanisms', 'summary', 'all'], default='all')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    cfg = read_config(args.config)
    out = args.run_dir.resolve()
    permitted = (args.project_root/cfg['artifact_root']).resolve()
    if out.parent != permitted:
        raise ValueError('run must be a fresh direct child of the configured artifact root')
    out.mkdir(parents=True, exist_ok=True)
    resolved = out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != cfg:
        raise ValueError('cannot change a retained run contract; use a fresh versioned run')
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    if args.stage in ('check', 'all'):
        write_json(out/'software_probe.json', software_probe(cfg))
    if args.stage == 'check':
        print(json.dumps(read_json(out/'software_probe.json')))
        return
    if not (out/'software_probe.json').exists() or read_json(out/'software_probe.json')['status'] != 'passed':
        raise ValueError('synthetic software probe must pass before measured stages')
    plan_path = out/'measured_plan.json'
    if not plan_path.exists():
        write_json(plan_path, make_plan(cfg, args.project_root))
    plan = read_json(plan_path)
    if args.stage == 'plan':
        print(json.dumps({k: plan[k] for k in ('expected_records', 'expected_windows', 'split')}))
        return
    workers = min(args.workers, cfg['resources']['max_workers'], len(os.sched_getaffinity(0)))
    if workers < 1:
        raise ValueError('workers must be positive')
    if args.stage in ('pilot', 'prepare', 'all'):
        records = plan['records']
        if args.stage == 'pilot':
            records = [next(r for r in records if r['dataset'] == ds) for ds in cfg['measured']['datasets']]
        results = run_tasks('prepare_worker', [(cfg, str(args.project_root), str(out), r) for r in records], out, workers)
        if args.stage == 'pilot':
            write_json(out/'pilot.json', results)
            return
    if args.stage in ('stability', 'all'):
        refs = read_json(args.project_root/cfg['attribution_run']/'measured_plan.json')['windows']
        run_tasks('stability_worker', [(cfg, str(args.project_root), str(out), r) for r in refs], out, workers)
    if args.stage in ('events', 'all'):
        write_json(out/'event_result.json', event_analysis(cfg, out, plan))
    if args.stage in ('hardware', 'all'):
        write_json(out/'hardware_result.json', hardware_analysis(cfg, out, plan))
    if args.stage in ('mechanisms', 'all'):
        if not (out/'mechanism_calibration.json').exists():
            calibrate_mechanisms(cfg, out)
        specs = [dict(repeat=i) for i in range(cfg['mechanisms']['evaluation_identities'])]
        run_tasks('mechanism_worker', [(cfg, str(args.project_root), str(out), s) for s in specs], out, workers)
    if args.stage in ('summary', 'all'):
        print(json.dumps(serial(summarize(cfg, out, plan, args.project_root)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
