#!/usr/bin/env python3
"""Bounded overnight N1--N7 diagnostics; ssm_next.md owns the experiment design.

One frozen task table, one process pool, per-cell alarm, memory tokens, and an
eight-hour termination budget. Result denominators come from registered cells,
including failures. This entry never invokes a teacher admission or tokenizer.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import scipy
import yaml
from scipy.integrate import solve_ivp
from scipy.signal import butter, sosfiltfilt, periodogram
from scipy.ndimage import gaussian_filter1d

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('SSM_PROJECT_ROOT', str(CODE_ROOT))).resolve()
sys.path.insert(0, str(CODE_ROOT))
from experiments import evaluate_step5 as step5
from experiments import evaluate_step5_observation_diagnostic as diagnostic
from experiments import evaluate_step5_observation_repair as repair
from src.inference import t3a_balloon_robust_ssm as core
from src.inference import t3a_balloon_joint_ssm as joint
from src.inference import balloon_trajectory_map as batch_map

# Frozen source may live below the run directory. Native paths still resolve
# through the single existing training-only loader, against the original repo.
step5.ROOT = diagnostic.ROOT = CODE_ROOT
repair.ROOT = ROOT
DEFAULT_CONFIG = CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml'
FAMILIES = ('N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7')
MODALITIES = ('EEG', 'HbO', 'HbR')
BASE = dict(id='baseline', eeg='E0', noise=1., gain=1., sigma_h=1., sigma_r=1., w=0., g=0.)
OUTER_MASKS = ('full', 'EEG_only', 'fNIRS_only', 'all_missing',
    'center_EEG', 'center_EEG_own', 'center_EEG_template', 'center_EEG_pairing', 'center_EEG_shift',
    'center_fNIRS', 'center_fNIRS_own', 'center_fNIRS_template', 'center_fNIRS_pairing', 'center_fNIRS_shift')
STATUSES = {'completed', 'data_unavailable', 'not_implemented', 'failed_domain',
            'failed_numerical', 'failed_contract', 'timeout', 'not_started_budget'}
difference_noise_constant = lru_cache(maxsize=4)(step5.student_difference_mad)


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(serial(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple, np.ndarray))
                         else serial(v) for k, v in row.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(stream.getvalue())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['schema'] not in ('ssm_overnight_v1', 'ssm_overnight_v2') or cfg['scope'] != 'bounded_training_only_exploratory_diagnostics' or
        cfg['subjects'] != ['subject_01', 'subject_09', 'subject_18'] or
        cfg['sessions'] != ['session_01', 'session_03', 'session_05'] or
        cfg['excluded_original_ma_positions'] != [4, 9] or cfg['outer_folds'] != 4 or cfg['inner_folds'] != 3 or
        cfg['fixed_w'] != [0., -.5] or cfg['teacher_qualification'] != 'none' or cfg['tokenizer_training'] is not False):
        raise ValueError('overnight data/model/interpretation contract drift')
    if (cfg['N3']['noise_multipliers'] != [.5, 1., 2., 4.] or
        cfg['N5']['gain_grid'] != [.5, .75, 1., 1.5, 2.] or
        cfg['N4']['candidates'] != ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'] or
        cfg['N6']['process_multipliers'] != [[1., 1.], [.25, 1.], [2., 1.], [1., .5], [1., 2.]]):
        raise ValueError('frozen one-factor candidate panel drift')
    if (cfg['schema'] == 'ssm_overnight_v2') != ('N7' in cfg):
        raise ValueError('N7 requires the versioned overnight v2 contract')
    if 'N7' in cfg:
        n7 = cfg['N7']
        if (n7['g_grid'] != [-.6, -.3, 0., .3, .6] or
            n7['w_grid'] != [-.5, -.25, 0., .25, .5] or
            n7['rules'] != ['GW', 'W_only', 'G_only'] or n7['surface_w_points'] != 9 or
            n7['synthetic_truth_conditions'] != ['true_g', 'true_w', 'true_gw', 'measurement_gain'] or
            n7['truth_g_values'] != [-.3, .3] or n7['truth_w_values'] != [-.25, .25] or
            n7['truth_observation_gains'] != [.75, 1.5]):
            raise ValueError('frozen G-W candidate/truth panel drift')
    b = cfg['budget']
    if not (0 < b['hours'] <= 8 and 1 <= b['max_workers'] <= 16 and 0 < b['memory_fraction'] <= .6 and
            0 < b['cell_timeout_seconds'] <= 900 and 0 < b['map_timeout_seconds'] <= 1800 and
            1 <= b['map_max_nfev'] <= 200):
        raise ValueError('budget exceeds bounded overnight contract')
    dc, base, measured, metadata = diagnostic.load_config(CODE_ROOT/cfg['diagnostic_config'])
    if (base['measured']['heldout_trial_positions'] != cfg['excluded_original_ma_positions'] or
            base['measured']['sessions'] != cfg['sessions']):
        raise ValueError('owning training boundary differs from suite')
    return cfg, dc, base, measured, metadata


def candidates(cfg, family):
    if family == 'N3':
        return [dict(BASE, id='baseline' if n == 1 else f'noise_{n:g}', noise=n) for n in cfg['N3']['noise_multipliers']]
    if family == 'N4':
        return [dict(BASE, id='baseline' if e == 'E0' else e, eeg=e) for e in cfg['N4']['candidates']]
    if family == 'N5':
        return [dict(BASE, id='baseline' if g == 1 else f'gain_{g:g}', gain=g) for g in cfg['N5']['gain_grid']]
    if family == 'N6':
        return [dict(BASE, id='baseline' if (h, r) == (1, 1) else f'process_h{h:g}_r{r:g}', sigma_h=h, sigma_r=r)
                for h, r in cfg['N6']['process_multipliers']]
    if family == 'N7':
        panel = [dict(BASE, id='baseline' if g == w == 0 else f'g{g:g}_w{w:g}', g=g, w=w)
                 for g in cfg['N7']['g_grid'] for w in cfg['N7']['w_grid']]
        # Stable ties retain the simpler reference; no score-dependent ordering.
        return sorted(panel, key=lambda c: (c['id'] != 'baseline', c['g'], c['w']))
    return [BASE]


def folds(identities, outer):
    train = [i for i, t in enumerate(identities) if t['training_ordinal'] % 4 != outer]
    validation = [i for i in range(len(identities)) if i not in train]
    inner = []
    for fold in range(3):
        ordinal = {i: j for session in sorted({identities[i]['session'] for i in train})
                   for j, i in enumerate([k for k in train if identities[k]['session'] == session])}
        fit = [i for i in train if ordinal[i] % 3 != fold]
        val = [i for i in train if i not in fit]
        if set(fit) & set(val) or set(fit+val) != set(train) or set(val) & set(validation):
            raise ValueError('inner/outer fold intersection')
        inner.append(dict(train=fit, validation=val))
    return dict(train=train, validation=validation, inner=inner)


def validate_identities(identities, cfg, subject):
    expected = {(s, p) for s in cfg['sessions'] for p in range(10) if p not in cfg['excluded_original_ma_positions']}
    if (subject not in cfg['subjects'] or len(identities) != 24 or
        any(t['subject'] != subject for t in identities) or
        {(t['session'], t['original_ma_trial_position']) for t in identities} != expected or
        len({t['sample_id'] for t in identities}) != 24):
        raise ValueError('scope identity boundary mismatch before signal access')
    for session in cfg['sessions']:
        rows = [t for t in identities if t['session'] == session]
        if [t['training_ordinal'] for t in rows] != list(range(8)):
            raise ValueError('surviving training ordinal differs from original trial position')
    for outer in range(4):
        split = folds(identities, outer)
        for session in cfg['sessions']:
            if (sum(identities[i]['session'] == session for i in split['train']) != 6 or
                sum(identities[i]['session'] == session for i in split['validation']) != 2):
                raise ValueError('outer per-session counts differ from 6/2')


def scope_inventory(cfg, base, metadata):
    """Canonical event metadata only. No signal reader before this validation."""
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    index = CleanPhysiologyCacheIndex(ROOT/metadata['data']['cache_root'])
    inventory = {}
    for subject in cfg['subjects']:
        records = sorted((r for r in index.records if r.dataset_id == base['measured']['dataset_id'] and
            r.canonical_subject_id == subject and r.base_record_id in cfg['sessions']), key=lambda r: r.base_record_id)
        if [r.base_record_id for r in records] != cfg['sessions']:
            raise ValueError('missing admitted session inventory')
        rows = []
        for record in records:
            for ordinal, (position, event) in enumerate(diagnostic.training_events(index.events_by_join_key[record.join_key], base)):
                rows.append(dict(subject=subject, session=record.base_record_id, training_ordinal=ordinal,
                    original_ma_trial_position=position, event_index=int(event['event_index']),
                    sample_id=f"{record.join_key}|event={event['event_index']}|offset=-5.0|duration=30.0"))
        validate_identities(rows, cfg, subject)
        inventory[subject] = rows
    if len({t['sample_id'] for rows in inventory.values() for t in rows}) != 72:
        raise ValueError('expected exactly 72 unique training trials')
    return inventory


def native_eeg_feature(values, band, hidden=False):
    values = np.array(values, dtype=float, copy=True)
    if values.ndim == 1:
        values = values[:, None]
    count = len(values)//50
    hidden_mask = np.zeros(count, dtype=bool)
    if hidden:
        _, hidden_mask, _ = step5.masked_input(np.zeros((count, 3)), 'center_EEG', 16)
        native_hidden = np.repeat(hidden_mask, 50)
        clock = np.arange(len(values))
        for col in range(values.shape[1]):
            values[native_hidden, col] = np.interp(clock[native_hidden], clock[~native_hidden], values[~native_hidden, col])
    filtered = sosfiltfilt(butter(4, band, btype='bandpass', fs=200., output='sos'), values, axis=0)
    result = np.log(np.maximum(np.mean(filtered.reshape(count, 50, values.shape[1])**2, axis=1), 1e-12))
    result[hidden_mask] = np.nan
    return result


def eog_coefficients(raw, auxiliary, training):
    if auxiliary.shape[-1] == 0:
        raise FileNotFoundError('E1 NOT_AVAILABLE: admitted native source has no EOG references')
    x = np.concatenate(auxiliary[training])
    y = np.concatenate(raw[training])
    center = x.mean(axis=0)
    design = x-center
    coefficients = np.linalg.lstsq(design, y-y.mean(axis=0), rcond=1e-10)[0]
    return center, coefficients


def cleaned_eeg(raw, eog, center, coefficients, hidden=False):
    raw, eog = np.array(raw, copy=True), np.array(eog, copy=True)
    if hidden:
        count = len(raw)//50
        _, missing, _ = step5.masked_input(np.zeros((count, 3)), 'center_EEG', 16)
        mask = np.repeat(missing, 50)
        clock = np.arange(len(raw))
        for values in (raw, eog):
            for j in range(values.shape[1]):
                values[mask, j] = np.interp(clock[mask], clock[~mask], values[~mask, j])
    return raw-(eog-center)@coefficients


def prepare_subject(run_dir, cfg, dc, base, measured, metadata, subject, expected):
    trials, detail = diagnostic.load_training_subject(subject, dc, base, measured, metadata, retain_native=True)
    validate_identities(detail['trials'], cfg, subject)
    if [t['sample_id'] for t in detail['trials']] != [t['sample_id'] for t in expected]:
        raise ValueError('native preparation disagrees with frozen scope inventory')
    arrays = dict(raw_eeg=np.array([t['native_eeg'] for t in trials]),
                  raw_eog=np.array([t['native_eog'] for t in trials]),
                  eligible=np.array([t['eligible'] for t in trials]))
    for view in ('target', 'center_EEG', 'center_fNIRS'):
        for key in ('eeg_log_power', 'fnirs'):
            arrays[view+'_'+key] = np.array([t['views'][view][key] for t in trials])
    directory = Path(run_dir)/'prepared'
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/f'{subject}.npz', **arrays)
    atomic_json(directory/f'{subject}.json', detail)
    return dict(subject=subject, trials=24, native_eog_channels=len(detail['eog_channels']),
                original_heldout_trials_processed=0)


def split_name(outer, inner=None):
    return 'common' if outer is None else f'o{outer}'+('' if inner is None else f'i{inner}')


def projection_path(run_dir, subject, outer, inner=None, eeg='E0'):
    return Path(run_dir)/'prepared'/f'{subject}_{split_name(outer, inner)}_{eeg}'


def local_channels(detail, pair, metadata, fallback='F3'):
    """Select by metadata distance only, never by EEG/fNIRS response scores."""
    from src.data.unified_physiology import ChannelGeometryIndex
    geometry = ChannelGeometryIndex(ROOT/metadata['data']['cache_root'])
    rows = [r for r in geometry.rows if r.get('dataset_id') == 'eeg_fnirs_single_trial' and
            r.get('canonical_subject_id') == detail['subject']]
    pair_label = detail['fnirs_pairs'][pair]
    pair_rows = [r for r in rows if r.get('modality') == 'fnirs' and r.get('channel_name') == pair_label]
    for target in pair_rows:
        coordinate = [target.get(k) for k in ('x', 'y', 'z')]
        if None in coordinate or not np.isfinite(coordinate).all():
            continue
        matched = []
        for index, name in enumerate(detail['eeg_channels']):
            available = [r for r in rows if r.get('modality') == 'eeg' and r.get('channel_name') == name and
                r.get('coordinate_system') == target.get('coordinate_system') and
                r.get('coordinate_units') == target.get('coordinate_units') and
                all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x', 'y', 'z'))]
            if len(available) == 1:
                point = [available[0][k] for k in ('x', 'y', 'z')]
                matched.append((float(np.linalg.norm(np.asarray(point)-coordinate)), name, index, available[0]))
        if matched:
            nearest = sorted(matched, key=lambda value: (value[0], value[1]))[:3]
            return [v[2] for v in nearest], dict(status='native_montage_nearest_eeg', pair=pair_label,
                channels=[v[1] for v in nearest], distances=[v[0] for v in nearest],
                coordinate_system=target['coordinate_system'], coordinate_units=target['coordinate_units'],
                fnirs_source=target.get('source_file'), eeg_sources=[v[3].get('source_file') for v in nearest],
                selection='nearest up to three available scalp EEG locations; no response correlation')
    if detail['eeg_channels'].count(fallback) != 1:
        raise FileNotFoundError('native montage mapping and declared F3 fallback unavailable')
    return [detail['eeg_channels'].index(fallback)], dict(status='anatomical_mapping_unverified', pair=pair_label,
        channels=[fallback], reason='no compatible native EEG/fNIRS coordinates')


def prepare_projection(run_dir, cfg, base, measured, subject, outer, inner=None, eeg='E0'):
    """A fit-fold owns all learned objects; common gauge is only for N5 curves."""
    prefix = projection_path(run_dir, subject, outer, inner, eeg)
    if prefix.with_suffix('.json').exists():
        return read_json(prefix.with_suffix('.json'))
    detail = read_json(Path(run_dir)/'prepared'/f'{subject}.json')
    identities = detail['trials']
    split = dict(train=list(range(24)), validation=list(range(24))) if outer is None else folds(identities, outer)
    if inner is not None:
        split = split['inner'][inner]
    train, val = split['train'], split['validation']
    with np.load(Path(run_dir)/'prepared'/f'{subject}.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    names = detail['eeg_channels']
    indices = list(range(len(names)))
    if eeg == 'E2':
        excluded = set(n.casefold() for n in cfg['N4']['excluded_frontal_channels'])
        indices = [i for i, name in enumerate(names) if name.casefold() not in excluded]
        if not indices:
            raise FileNotFoundError('fixed nonfrontal subset unavailable')
    if eeg in ('E3', 'E4', 'E5'):
        baseline = read_json(projection_path(run_dir, subject, outer, inner, 'E0').with_suffix('.json'))
        _, _, _, _, metadata = load_config(Path(run_dir)/'resolved_config.yaml')
        indices, anatomy = local_channels(detail, baseline['projection']['fnirs_pair'], metadata,
                                          cfg['N4']['fallback_local_channel'])
    regression = None
    if eeg == 'E1':
        regression = eog_coefficients(arrays['raw_eeg'], arrays['raw_eog'], train)
    features = {}
    for view in ('target', 'center_EEG', 'center_fNIRS'):
        eeg_values = arrays[view+'_eeg_log_power'][:, :, indices].copy()
        if eeg in ('E1', 'E4', 'E5'):
            constructed = []
            for i, raw in enumerate(arrays['raw_eeg']):
                if eeg == 'E1':
                    raw = cleaned_eeg(raw, arrays['raw_eog'][i], *regression, hidden=view == 'center_EEG')
                constructed.append(native_eeg_feature(raw[:, indices], [8., 13.] if eeg in ('E4', 'E5') else [1., 45.],
                                                       hidden=view == 'center_EEG'))
            eeg_values = np.array(constructed)
        features[view] = [dict(eeg_log_power=eeg_values[i], fnirs=arrays[view+'_fnirs'][i]) for i in range(24)]
    eligible = np.logical_and.reduce(arrays['eligible'][train])
    projection = step5.fit_measured_projection([features['target'][i] for i in train], base, measured, eligible)
    baseline_data = None
    if eeg != 'E0':
        baseline_path = projection_path(run_dir, subject, outer, inner, 'E0')
        baseline = read_json(baseline_path.with_suffix('.json'))
        for key in ('fnirs_pair', 'fnirs_factor', 'pair_scores', 'pair_eligible'):
            projection[key] = baseline['projection'][key]
        with np.load(baseline_path.with_suffix('.npz'), allow_pickle=False) as archive:
            baseline_data = {k: archive[k] for k in archive.files}
    targets = {}
    for view, rows in features.items():
        targets[view] = np.array([step5.apply_measured_projection(f, projection) for f in rows])
        if eeg == 'E5':
            targets[view][:, :, 0] *= -1
        if baseline_data is not None:
            np.testing.assert_array_equal(targets[view][:, :, 1:], baseline_data[view][:, :, 1:])
    normalizer = np.maximum(np.std(np.concatenate(targets['target'][train]), axis=0), 1e-12)
    constant = difference_noise_constant(base['model']['student_nu'])
    noise = np.maximum(step5.first_difference_noise(targets['target'][train], constant), base['model']['observation_scale'])
    if baseline_data is not None:
        normalizer[1:] = baseline_data['normalizer'][1:]
        noise[1:] = baseline_data['noise'][1:]
    targets.update(normalizer=normalizer, noise=noise, valid_pairs=arrays['eligible'][:, projection['fnirs_pair']])
    np.savez_compressed(prefix.with_suffix('.npz'), **targets)
    info = dict(subject=subject, outer=outer, inner=inner, eeg=eeg, train=train, validation=val,
        projection=projection, selected_eeg_channels=[names[i] for i in indices],
        anatomical_mapping=anatomy if eeg in ('E3', 'E4', 'E5') else dict(status='not_applicable'),
        regression=None if regression is None else dict(center=regression[0], coefficients=regression[1], training_trials=train),
        noise_scale=noise, normalization_sd=normalizer,
        coordinate_use='N5_common_coordinate_descriptive_only' if outer is None else 'nested_candidate_scoring',
        hbo_hbr_contract='canonical MBLL [HbO,HbR], common baseline and common scale; HbT=sum, no sign flip',
        native_eeg_reconstruction='NOT_SUPPORTED: log power and PCA are not invertible to voltage')
    atomic_json(prefix.with_suffix('.json'), info)
    return info


@lru_cache(maxsize=48)
def load_projection(run_dir, subject, outer, inner, eeg):
    prefix = projection_path(run_dir, subject, outer, inner, eeg)
    info = read_json(prefix.with_suffix('.json'))
    with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return info, arrays


def model(base, candidate, noise=None):
    config = copy.deepcopy(base)
    if noise is not None:
        config['model']['observation_scale'] = np.asarray(noise).tolist()
    g, w = candidate.get('g', 0.), candidate['w']
    if any(not np.isfinite(value) or not base['axes'][axis]['bounds'][0] <= value <= base['axes'][axis]['bounds'][1]
           for axis, value in (('G', g), ('W', w))):
        raise ValueError('G/W outside registered support; no clipping')
    reference = config['model']['reference']
    zeta = reference['kappa']/(2*np.sqrt(reference['gamma']))
    p, c = step5.multi_parameter_model(config, [g, w, zeta])
    scale = np.asarray(p.fixed.observation_scale).copy()
    scale[1:] *= candidate['noise']
    process = np.asarray(p.fixed.process_std).copy()
    process[0] *= candidate['sigma_r']
    process[1:] *= candidate['sigma_h']
    p = replace(p, fixed=replace(p.fixed, process_std=tuple(process), observation_scale=tuple(scale)))
    spec = core.BalloonObservationSpec(fnirs_gain=candidate['gain']).resolved(p.fixed)
    return p, c, spec


def input_for(arrays, info, identities, trial, mode):
    targets = arrays['target']
    y = targets[trial].copy()
    if mode == 'full':
        return y
    if mode in ('EEG_only', 'fNIRS_only', 'all_missing'):
        y[:, [1, 2] if mode == 'EEG_only' else [0] if mode == 'fNIRS_only' else [0, 1, 2]] = np.nan
        return y
    modality = 'EEG' if mode.startswith('center_EEG') else 'fNIRS'
    columns = [1, 2] if modality == 'EEG' else [0]
    y = arrays['center_'+modality][trial].copy()
    suffix = mode.removeprefix('center_'+modality)
    if suffix == '_own':
        y[:, columns] = np.nan
    elif suffix in ('_template', '_pairing'):
        peers = [i for i in info['train'] if identities[i]['session'] == identities[trial]['session'] and i != trial]
        if not peers:
            raise ValueError('no independent training template or donor')
        source = (targets[peers].mean(axis=0) if suffix == '_template' else
                  targets[peers[identities[trial]['training_ordinal'] % len(peers)]])
        y[:, columns] = source[:, columns]
    elif suffix == '_shift':
        y[:, columns] = np.roll(y[:, columns], len(y)//2, axis=0)
    return y


def residual_metrics(target, prediction, sd, mask=None):
    selected = np.ones(len(target), dtype=bool) if mask is None else mask
    error = prediction[selected]-target[selected]
    result = {}
    for i, name in enumerate(MODALITIES):
        e = error[:, i]
        if not len(e) or not np.isfinite(e).all():
            result[name] = None
            continue
        normed = e/sd[i]
        result[name] = dict(bias_coordinate=float(e.mean()), rmse_coordinate=float(np.sqrt(np.mean(e**2))),
            bias_sd=float(normed.mean()), nrmse=float(np.sqrt(np.mean(normed**2))),
            nmse=float(np.mean(normed**2)), mae_sd=float(np.mean(abs(normed))),
            absolute_residual_sd_quantiles=np.quantile(abs(normed), [.5, .9, .95]), samples=len(e))
    return result


def noise_structure(values):
    result = {}
    for i, name in enumerate(MODALITIES):
        v = values[:, i]
        if not np.isfinite(v).all():
            result[name] = None
            continue
        centered = v-v.mean()
        freq, power = periodogram(centered, fs=4.)
        acf = [1.]+[float(centered[:-lag]@centered[lag:]/max(np.linalg.norm(centered[:-lag])*np.linalg.norm(centered[lag:]), 1e-20))
                    for lag in range(1, min(21, len(v)))]
        result[name] = dict(bias=float(v.mean()), rms=float(np.sqrt(np.mean(v**2))), acf=acf,
            low_frequency_fraction=float(power[(freq > 0)&(freq <= .2)].sum()/max(power.sum(), 1e-20)))
    hb = values[:, 1:]
    result['HbO_HbR_correlation'] = correlation(hb[:, 0], hb[:, 1])
    return result


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if (np.isfinite(a).all() and np.isfinite(b).all() and
        np.std(a) > 1e-15 and np.std(b) > 1e-15) else None


def clean_metrics(truth, mean, variance=None):
    result = {}
    for i, label in enumerate(('r', 'clean_EEG', 'clean_HbO', 'clean_HbR')):
        e = mean[:, i]-truth[:, i]
        score = dict(nrmse=float(np.sqrt(np.mean(e**2))/max(np.std(truth[:, i]), 1e-12)),
                     correlation=correlation(truth[:, i], mean[:, i]), uncertainty='NOT_ESTIMATED')
        if variance is not None and np.isfinite(variance[:, i]).all() and np.all(variance[:, i] >= 0):
            width = 1.95996398454*np.sqrt(variance[:, i])
            score.update(coverage95=float(np.mean(abs(e) <= width)), mean_interval_width=float(np.mean(2*width)),
                         uncertainty='conditional_approximation')
        result[label] = score
    return result


def physical_valid(checks):
    return all(checks.get(k, False) for k in ('finite', 'positive_fvpq', 'oxygen_extraction_in_unit_interval',
                                             'absolute_hb_nonnegative', 'hbr_not_above_hbt'))


def failure(exc):
    if isinstance(exc, core.FlowDomainExit):
        return dict(status='failed_domain', error=repr(exc), flow_domain_exit=exc.diagnostic)
    if isinstance(exc, FileNotFoundError):
        return dict(status='data_unavailable', error=repr(exc))
    if isinstance(exc, (ValueError, AssertionError, KeyError)):
        return dict(status='failed_contract', error=repr(exc))
    if isinstance(exc, TimeoutError):
        return dict(status='timeout', error=repr(exc))
    return dict(status='failed_numerical', error=repr(exc), traceback=traceback.format_exc())


def save_fit(run_dir, cell_id, row_id, arrays, row):
    directory = Path(run_dir)/'cells'/cell_id
    directory.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        path = directory/(row_id+'.npz')
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        row['trajectory_path'] = str(path.relative_to(run_dir))
    atomic_json(directory/(row_id+'.json'), row)


def infer_row(run_dir, task, row_id, y, target, sd, p, c, spec, *, mode='full', truth=None):
    row = dict(row_id=row_id, mode=mode)
    try:
        fit = joint.smooth_balloon_joint(y, p, config=c, observation_spec=spec, quadrature_order=task.get('quadrature_order', 13))
        hidden = np.zeros(len(y), dtype=bool)
        hidden[(len(y)-16)//2:(len(y)+16)//2] = True
        row.update(status='completed', metrics=residual_metrics(target, fit.trajectory_mean, sd),
            center_metrics=residual_metrics(target, fit.trajectory_mean, sd, hidden),
            parameter_log_likelihood=fit.parameter_log_likelihood, physical_checks=fit.physical_checks,
            residual_sign='prediction_minus_target; one-step innovations use input_minus_prediction',
            native_residual_status='NOT_SUPPORTED: EEG voltage / optical intensity cannot be recovered through power/PCA/optics/motion processing',
            clean_interval_status='approximate_pointwise_solver_coordinate',
            noisy_interval_status='NOT_ESTIMATED: processed target shares native noise; no independent-noise shortcut',
            parameter_interval='NOT_ESTIMATED')
        if not physical_valid(fit.physical_checks):
            row.update(status='failed_domain', solver_completed=True,
                       error='posterior mean violates recorded physical constraints; trajectory retained')
        data = dict(r=fit.state_mean[:, 0], transformed_mean=fit.transformed_mean,
            clean_mean=fit.trajectory_mean, clean_variance=fit.state_posterior_variance,
            r_variance=fit.state_covariance[:, 0, 0], observation_mask=np.isfinite(y), input=y,
            target=target, normalization_sd=sd)
        if mode == 'full':
            predicted = []
            for m, cov in zip(fit.predicted_transformed_mean, fit.predicted_transformed_covariance):
                physical_mean, physical_cov = core.transformed_gaussian_moments(m, cov)
                h = core._observation_physical_matrix(p, spec)
                mean = core._observation_map_unchecked(physical_mean, p, spec)
                variance = np.diag(h@physical_cov@h.T)+fit.observation_variance
                predicted.append((y[len(predicted)]-mean)/np.sqrt(variance))
            row['innovation_structure'] = noise_structure(np.asarray(predicted))
            row['smoothing_residual_structure'] = noise_structure((target-fit.trajectory_mean)/sd)
            row['hbt'] = dict(bias_coordinate=float(np.mean(np.sum(fit.trajectory_mean[:, 1:]-target[:, 1:], axis=1))),
                             rmse_coordinate=float(np.sqrt(np.mean(np.sum(fit.trajectory_mean[:, 1:]-target[:, 1:], axis=1)**2))))
            data['standardized_innovations'] = np.asarray(predicted)
        if truth is not None:
            row['truth'] = clean_metrics(truth, np.column_stack((fit.state_mean[:, 0], fit.trajectory_mean)),
                np.column_stack((fit.state_covariance[:, 0, 0], fit.state_posterior_variance)))
        save_fit(run_dir, task['id'], row_id, data, row)
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            raise
        row.update(failure(exc))
        save_fit(run_dir, task['id'], row_id, None, row)
    return row


def task_id(*parts):
    return '__'.join(str(p).replace('.', 'p').replace('-', 'm') for p in parts)


def make_tasks(cfg, inventory):
    tasks = []
    families = FAMILIES if 'N7' in cfg else FAMILIES[:-1]

    def add(family, kind, parts, solves, layer=2, dependencies=(), **payload):
        task = dict(id=task_id(family, kind, *parts), family=family, kind=kind, layer=layer,
                    planned_solves=solves, dependencies=list(dependencies), **payload)
        tasks.append(task)
        return task['id']

    for subject, identities in inventory.items():
        for outer in range(4):
            split = folds(identities, outer)
            baseline_inner = []
            for inner in range(3):
                baseline_inner.append(add('N1', 'inner', [subject, outer, inner], 12,
                    subject=subject, outer=outer, inner=inner, candidate=BASE))
            baseline_outer = {}
            for i in split['validation']:
                for w in cfg['fixed_w']:
                    cell = add('N1', 'outer', [subject, outer, i, w], len(OUTER_MASKS),
                               subject=subject, outer=outer, trial=i, candidate=dict(BASE, w=w))
                    if w == 0:
                        baseline_outer[i] = cell
            for family in families[2:]:
                inner_cells = {}
                for candidate in candidates(cfg, family):
                    ids = baseline_inner.copy() if candidate['id'] == 'baseline' else [
                        add(family, 'inner', [subject, outer, inner, candidate['id']], 12,
                            subject=subject, outer=outer, inner=inner, candidate=candidate) for inner in range(3)]
                    inner_cells[candidate['id']] = ids
                for rule in cfg['N7']['rules'] if family == 'N7' else ('single_factor',):
                    allowed = {c['id'] for c in candidates(cfg, family)
                               if rule not in ('W_only', 'G_only') or
                               (c['g'] == 0 if rule == 'W_only' else c['w'] == 0)}
                    rule_cells = {name: ids for name, ids in inner_cells.items() if name in allowed}
                    suffix = [rule] if family == 'N7' else []
                    selection = add(family, 'select', [subject, outer, *suffix], 0, subject=subject, outer=outer,
                        rule=rule, inner_cells=rule_cells, dependencies=[v for values in rule_cells.values() for v in values])
                    for i in split['validation']:
                        add(family, 'selected_outer', [subject, outer, i, *suffix], len(OUTER_MASKS),
                            rule=rule, subject=subject, outer=outer, trial=i, selection=selection,
                            baseline_cell=baseline_outer[i], dependencies=[selection, baseline_outer[i]])
            # Same-fold linear baselines use their own inner ridge selection.
            for modality in ('EEG', 'fNIRS'):
                add('N1', 'linear', [subject, outer, modality], 0,
                    subject=subject, outer=outer, modality=modality)
            add('N3', 'noise_diagnostic', [subject, outer], 0, subject=subject, outer=outer)
            for i in split['validation']:
                add('N6', 'replay', [subject, outer, i], 0, subject=subject, outer=outer, trial=i,
                    baseline_cell=baseline_outer[i], dependencies=[baseline_outer[i]])
        for eeg in cfg['N4']['candidates']:
            add('N4', 'artifact', [subject, eeg], 5, subject=subject, eeg=eeg, outer=0)
        grid = [(float(w), 1.) for w in np.linspace(-.5, .5, cfg['N5']['w_points'])]
        grid += [(float(w), float(g)) for w in np.linspace(-.5, .5, cfg['N5']['surface_w_points'])
                 for g in cfg['N5']['gain_grid'] if g != 1]
        curve_cells = []
        for w, gain in grid:
            curve_cells.append(add('N5', 'curve', [subject, w, gain], 24, layer=2 if gain == 1 else 3,
                subject=subject, candidate=dict(BASE, w=w, gain=gain)))
        add('N5', 'session_summary', [subject], 0, layer=3, subject=subject, curve_cells=curve_cells, dependencies=curve_cells)
        if 'N7' in cfg:
            gw_cells = []
            for w in np.linspace(-.5, .5, cfg['N7']['surface_w_points']):
                for g in cfg['N7']['g_grid']:
                    if g == 0:
                        identifier = task_id('N5', 'curve', subject, float(w), 1.)
                    else:
                        identifier = add('N7', 'curve', [subject, float(w), g], 24, layer=3,
                            subject=subject, candidate=dict(BASE, g=g, w=float(w)))
                    gw_cells.append(identifier)
            add('N7', 'session_summary', [subject], 0, layer=3, subject=subject,
                curve_cells=gw_cells, dependencies=gw_cells)
    for subject, trial in cfg['N6']['replay_training_indices']:
        for w in cfg['fixed_w']:
            add('N6', 'failure_trace', [subject, trial, w], 1, layer=1, subject=subject, trial=trial, w=w)
    for law in cfg['N2']['laws']:
        for replicate in range(cfg['N2']['trials_per_law']):
            for variant in cfg['N2']['variants']:
                masks = cfg['N2']['masks']+(['whole_EEG', 'whole_fNIRS'] if replicate < cfg['N2']['whole_mask_trial_count'] else [])
                for mode in masks:
                    for w in cfg['fixed_w']:
                        for solver in ('O0', 'O1', 'O2'):
                            rank_check = replicate < 2 and variant == 'combined' and mode == 'full' and w == 0 and solver != 'O0'
                            add('N2', 'temporal', [law, replicate, variant, mode, w, solver], 3 if rank_check else 1,
                                law=law, replicate=replicate, variant=variant, mode=mode, w=w, solver=solver,
                                seed=cfg['N2']['seed_start'][law]+replicate, rank_check=rank_check)
    add('N2', 'measured_map', [], 0, layer=1,
        reason='NOT_IMPLEMENTED: native helper does not expose pre-linear EEG-feature/Hb-concentration boundary; no substitute loader')
    # Freeze truth/noise streams before any candidate results. Every invalid
    # draw remains attached to its original seed; no redraw or successful subset.
    for stream, seed in cfg['synthetic']['streams'].items():
        for condition_index, condition in enumerate(cfg['synthetic']['conditions']):
            for replicate in range(cfg['synthetic']['replicates_per_condition']):
                for family in families:
                    if family == 'N2' or family == 'N4':
                        continue  # N4 has native spatial/EOG contamination, not a fictional spatial generator.
                    panel = candidates(cfg, family)
                    if family != 'N1':
                        panel = [c for c in panel if c['id'] != 'baseline']
                    else:
                        panel = [BASE, dict(BASE, id='legacy_w', w=-.5)]
                    for candidate in panel:
                        add(family, 'synthetic', [stream, condition, replicate, candidate['id']], 3,
                            layer=3 if family == 'N7' and condition != 'matched_student_t' else 2,
                            stream=stream, condition=condition, replicate=replicate, candidate=candidate,
                            seed=seed+condition_index*100+replicate)
        if 'N7' in cfg:
            for index, condition in enumerate(cfg['N7']['synthetic_truth_conditions']):
                for replicate in range(cfg['synthetic']['replicates_per_condition']):
                    panel = candidates(cfg, 'N7')+[dict(c, id='observation_'+c['id'])
                        for c in candidates(cfg, 'N5') if c['id'] != 'baseline']
                    for candidate in panel:
                        add('N7', 'synthetic', [stream, condition, replicate, candidate['id']], 3, layer=3,
                            stream=stream, condition=condition, replicate=replicate, candidate=candidate,
                            seed=seed+10000+index*100+replicate)
    # First-layer coverage contains the earliest fixed identity in each family,
    # independent of scores. All remaining core work is family round-robin.
    for family in families:
        first = next(t for t in tasks if t['family'] == family and t['kind'] in ('inner', 'temporal'))
        first['layer'] = 1
    used = Counter()
    for task in sorted(tasks, key=lambda t: t['layer']):
        family = task['family']
        if task['layer'] < 3 and used[family]+task['planned_solves'] > cfg['budget']['core_solve_caps'][family]:
            task['layer'] = 3
        if task['layer'] < 3:
            used[family] += task['planned_solves']
    # Selection dependencies cannot be left behind their dependent outer cells.
    by_id = {t['id']: t for t in tasks}
    for task in tasks:
        task['layer'] = max([task['layer']]+[by_id[d]['layer'] for d in task['dependencies']])
    ordered = []
    for layer in (1, 2, 3):
        queues = {family: deque(t for t in tasks if t['family'] == family and t['layer'] == layer) for family in families}
        while any(queues.values()):
            for family in families:
                if queues[family]:
                    ordered.append(queues[family].popleft())
    for position, task in enumerate(ordered):
        task['queue_position'] = position
        task['timeout_seconds'] = cfg['budget']['map_timeout_seconds'] if task.get('solver') == 'O2' else cfg['budget']['cell_timeout_seconds']
    if len({t['id'] for t in ordered}) != len(ordered):
        raise ValueError('duplicate task identity')
    return ordered


def persist_tasks(run_dir, tasks):
    write_csv(Path(run_dir)/'task_table.csv', [dict(task_id=t['id'], family=t['family'], kind=t['kind'],
        layer=t['layer'], queue_position=t['queue_position'], planned_solves=t['planned_solves'],
        timeout_seconds=t['timeout_seconds'], payload=t) for t in tasks])


def read_tasks(run_dir):
    with (Path(run_dir)/'task_table.csv').open() as stream:
        return [json.loads(r['payload']) for r in csv.DictReader(stream)]


def cell_result(run_dir, identifier):
    path = Path(run_dir)/'cells'/identifier/'result.json'
    return read_json(path) if path.exists() else dict(status='not_started_budget', rows=[])


def measured_cell(run_dir, cfg, base, task):
    candidate = task['candidate']
    info, arrays = load_projection(str(run_dir), task['subject'], task['outer'], task.get('inner'), candidate['eeg'])
    identities = read_json(Path(run_dir)/'prepared'/f"{task['subject']}.json")['trials']
    p, c, spec = model(base, candidate, arrays['noise'])
    indices = info['validation'] if task['kind'] == 'inner' else [task['trial']]
    modes = ('center_EEG', 'center_fNIRS') if task['kind'] == 'inner' else OUTER_MASKS
    rows = []
    for i in indices:
        for mode in modes:
            row_id = task_id(i, mode)
            if not arrays['valid_pairs'][i]:
                row = dict(row_id=row_id, mode=mode, status='failed_contract', error='training-selected fNIRS pair invalid; no replacement')
                save_fit(run_dir, task['id'], row_id, None, row)
            else:
                y = input_for(arrays, info, identities, i, mode)
                row = infer_row(run_dir, task, row_id, y, arrays['target'][i], arrays['normalizer'], p, c, spec, mode=mode)
            row.update(trial=i, subject=task['subject'], session=identities[i]['session'],
                       sample_id=identities[i]['sample_id'], candidate=candidate['id'], outer=task['outer'])
            save_fit(run_dir, task['id'], row_id, None, row)
            rows.append(row)
    return dict(status='completed', rows=rows, actual_solves=len(indices)*len(modes),
                candidate=candidate, fit_train=info['train'], validation=indices,
                coordinate_use=info['coordinate_use'], noise_scale=arrays['noise'], normalization_sd=arrays['normalizer'])


def center_risk(rows):
    values = defaultdict(list)
    for row in rows:
        if row['status'] != 'completed':
            return None
        columns = ('EEG',) if row['mode'] == 'center_EEG' else ('HbO', 'HbR')
        for col in columns:
            metric = row['center_metrics'][col]
            if not metric or metric['nmse'] is None:
                return None
            values[col].append(metric['nmse'])
    if any(not values[col] for col in MODALITIES):
        return None
    nmse = np.array([np.mean(values[col]) for col in MODALITIES])
    return dict(nmse=nmse.tolist(), nrmse=np.sqrt(nmse).tolist(), B=float(nmse@[.5, .25, .25]),
                fnirs=float(nmse[1:].mean()))


def select_candidate(run_dir, cfg, task):
    results = {}
    for candidate, ids in task['inner_cells'].items():
        blocks = [cell_result(run_dir, identifier) for identifier in ids]
        rows = [r for block in blocks for r in block.get('rows', [])]
        risk = center_risk(rows) if len(rows) == 36 and all(b['status'] == 'completed' for b in blocks) else None
        results[candidate] = dict(risk=risk, expected_fits=36, completed_fits=sum(r['status'] == 'completed' for r in rows),
                                  missing_or_failed=36-sum(r['status'] == 'completed' for r in rows))
    baseline = results['baseline']['risk']
    if baseline is None:
        return dict(status='failed_contract', error='baseline inner identity set incomplete; selection not defined', candidates=results, rows=[])
    acceptable = []
    for candidate, value in results.items():
        risk = value['risk']
        columns = slice(1, 3) if task['family'] == 'N4' else slice(0, 3)
        if risk is not None and np.all(np.asarray(risk['nrmse'])[columns] <=
                np.asarray(baseline['nrmse'])[columns]+cfg['selection']['maximum_modality_nrmse_degradation']):
            acceptable.append(candidate)
    endpoint = 'fnirs' if task['family'] == 'N4' else 'B'
    selected = min(acceptable, key=lambda name: (results[name]['risk'][endpoint], list(results).index(name)))
    candidate = next(c for c in candidates(cfg, task['family']) if c['id'] == selected)
    return dict(status='completed', selected=candidate, candidates=results, endpoint=endpoint,
                selection_source='inner folds only; paired-null and truth screening deferred to complete outer evidence', rows=[])


def selected_outer(run_dir, cfg, base, task):
    selection = cell_result(run_dir, task['selection'])
    if selection['status'] != 'completed':
        return dict(status='failed_contract', error='inner selection incomplete', rows=[])
    candidate = selection['selected']
    if candidate['id'] == 'baseline':
        result = copy.deepcopy(cell_result(run_dir, task['baseline_cell']))
        result.update(reused_baseline_cell=task['baseline_cell'], actual_solves=0, selected_candidate=candidate)
        return result
    result = measured_cell(run_dir, cfg, base, dict(task, kind='outer', candidate=candidate))
    result['selected_candidate'] = candidate
    return result


def linear_cell(run_dir, cfg, dc, task):
    subject, outer, modality = task['subject'], task['outer'], task['modality']
    identities = read_json(Path(run_dir)/'prepared'/f'{subject}.json')['trials']

    def examples(inner):
        info, arrays = load_projection(str(run_dir), subject, outer, inner, 'E0')
        groups = {}
        for role, indices in [('train', info['train']), ('validation', info['validation'])]:
            rows = []
            for i in indices:
                peers = [j for j in info['train'] if j != i and identities[j]['session'] == identities[i]['session']]
                template = arrays['target'][peers].mean(axis=0)
                y = arrays['center_'+modality][i]
                basic, full, hidden, cols = diagnostic.linear_features(y, template, modality, dc)
                donor = arrays['target'][peers[identities[i]['training_ordinal'] % len(peers)]]
                _, pairing, _, _ = diagnostic.linear_features(y, template, modality, dc, donor)
                _, shift, _, _ = diagnostic.linear_features(y, template, modality, dc, np.roll(y, len(y)//2, axis=0))
                rows.append(dict(trial=i, basic=basic, joint=full, pairing=pairing, shift=shift,
                                 target=arrays['target'][i][hidden][:, cols], sd=arrays['normalizer'][cols]))
            groups[role] = rows
        return groups

    alphas = dc['linear']['ridge_alphas']
    losses = {kind: np.zeros(len(alphas)) for kind in ('basic', 'joint')}
    for inner in range(3):
        groups = examples(inner)
        y = np.concatenate([r['target'] for r in groups['train']])
        for kind in losses:
            x = np.concatenate([r[kind] for r in groups['train']])
            for a, alpha in enumerate(alphas):
                fit = diagnostic.ridge_fit(x, y, alpha)
                losses[kind][a] += np.mean([np.mean(((diagnostic.ridge_predict(fit, r[kind])-r['target'])/r['sd'])**2)
                                           for r in groups['validation']])
    groups = examples(None)
    selected = {kind: alphas[int(np.argmin(loss))] for kind, loss in losses.items()}
    models = {kind: diagnostic.ridge_fit(np.concatenate([r[kind] for r in groups['train']]),
        np.concatenate([r['target'] for r in groups['train']]), alpha) for kind, alpha in selected.items()}
    rows = []
    for item in groups['validation']:
        scores = {kind: float(np.mean(((diagnostic.ridge_predict(models['basic' if kind == 'basic' else 'joint'], item[kind])-
                 item['target'])/item['sd'])**2)) for kind in ('basic', 'joint', 'pairing', 'shift')}
        rows.append(dict(trial=item['trial'], subject=subject, session=identities[item['trial']]['session'],
                         sample_id=identities[item['trial']]['sample_id'], status='completed', nmse=scores,
                         increment={k: v-scores['joint'] for k, v in scores.items() if k != 'joint'}))
    return dict(status='completed', selected_alphas=selected, inner_losses=losses, rows=rows, actual_solves=0)


def generated_trial(base, law, seed, steps=64, *, truth_g=0., truth_w=0., truth_gain=1.):
    cfg = copy.deepcopy(base)
    cfg['model']['steps'] = steps
    parameters, numerics, _ = model(base, dict(BASE, g=truth_g, w=truth_w))
    if truth_g != 0 or truth_w != 0:
        cfg['model']['reference'].update(beta=parameters.fixed.neurovascular_gain,
            gamma=parameters.fixed.gamma, kappa=parameters.free.kappa)
    if law == 'linearized_gaussian':
        return repair.generate_linear_gaussian(cfg, steps, seed)
    value = step5.localization.generate_matched(cfg, 'W', 0., seed)
    for t, state in enumerate(value['transformed_states'][:-1]):
        try:
            core._require_flow_drift_domain(state, parameters, numerics.dt)
        except core.FlowDomainExit as exc:
            exc.diagnostic.update(generation_transition_index=t+1, seed=seed,
                failure_stage='generating_truth_support; fixed seed retained, no redraw')
            raise
    if law == 'nonlinear_gaussian':
        p, _, _ = model(base, BASE)
        variance = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
        value['observations'] = value['clean']+np.random.default_rng(seed+400000).normal(size=value['clean'].shape)*np.sqrt(variance)
    if truth_gain != 1:
        # A mean-only measurement mismatch: retain exactly the same noise draw.
        noise = value['observations']-value['clean']
        value['clean'][:, 1:] *= truth_gain
        value['observations'] = value['clean']+noise
    return value


@lru_cache(maxsize=12)
def bridge_operator(steps, variant, diagnostic_config):
    dc = yaml.safe_load(Path(diagnostic_config).read_text())
    return repair.trajectory_operator(steps, variant, dc)


def temporal_cell(run_dir, cfg, base, task):
    generated = generated_trial(base, task['law'], task['seed'], cfg['synthetic']['steps'])
    count = len(generated['clean'])
    p, c, spec = model(base, dict(BASE, w=task['w']))
    operator = bridge_operator(count, task['variant'], str(CODE_ROOT/cfg['diagnostic_config']))
    mask = repair.bridge_mask(count, task['mode'], cfg['center_steps'])
    operator = operator.with_visible_interpolation(mask, mask)
    y = operator.apply(generated['observations'], mask)
    variance = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
    truth = np.column_stack((generated['states'][:, 0], generated['clean']))
    uncertainty = None
    if task['solver'] == 'O0':
        fit = joint.smooth_balloon_joint(y, p, config=c, quadrature_order=cfg['quadrature_order'])
        mean = np.column_stack((fit.state_mean[:, 0], fit.trajectory_mean))
        uncertainty = np.column_stack((fit.state_covariance[:, 0, 0], fit.state_posterior_variance))
        diagnostics = dict(parameter_log_likelihood=fit.parameter_log_likelihood, physical_checks=fit.physical_checks)
        z = fit.transformed_mean
    elif task['solver'] == 'O1':
        fit = joint.smooth_balloon_trajectory_reference(y, p, config=c, trajectory_spec=operator,
            noise_variance=variance, input_mask=mask, rank_rtol=cfg['N2']['rank_rtol'])
        z = fit['transformed_mean']
        mean = np.column_stack((z[:, 0], fit['canonical_clean_mean']))
        uncertainty = np.column_stack((np.diag(fit['transformed_covariance'])[::6],
            np.diag(fit['canonical_clean_covariance']).reshape(count, 3)))
        diagnostics = {k: fit[k] for k in ('retained_rank', 'observed_coordinates', 'discarded_singular_values',
            'support_residual_norm', 'parameter_log_likelihood', 'approximation')}
        diagnostics['physical_checks_at_transformed_mean'] = core.run_physical_checks(
            np.array([core.transformed_to_physical(row) for row in z]), p)
    else:
        checks = read_json(Path(run_dir)/'preflight.json')
        if not checks['O2']['passed']:
            return dict(status='failed_contract', error='O2 engineering preflight failed', rows=[])
        fit = batch_map.smooth_balloon_trajectory_map(y, p, config=c, trajectory_spec=operator,
            noise_variance=variance, input_mask=mask, max_nfev=cfg['budget']['map_max_nfev'],
            rank_rtol=cfg['N2']['rank_rtol'])
        diagnostics = {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)}
        if fit['status'] != 'completed':
            return dict(**diagnostics, rows=[], actual_solves=1)
        z = fit['transformed_mean']
        mean = np.column_stack((z[:, 0], fit['canonical_clean_mean']))
    row = {**diagnostics, 'status': 'completed', 'mode': task['mode'], 'truth': clean_metrics(truth, mean, uncertainty)}
    target = operator.apply(generated['clean'], mask)
    output = operator.matrix(mask)[0]@mean[:, 1:].ravel() if task['solver'] != 'O0' else mean[:, 1:].ravel()
    selected = np.isfinite(target)
    row['processed_visible_rmse'] = [float(np.sqrt(np.mean((output.reshape(count, 3)[selected[:, m], m]-target[selected[:, m], m])**2)))
                                    if selected[:, m].any() else None for m in range(3)]
    hidden_times = ~mask.all(axis=1)
    row['hidden_truth'] = clean_metrics(truth[hidden_times], mean[hidden_times],
        None if uncertainty is None else uncertainty[hidden_times]) if hidden_times.any() else None
    row['noisy_prediction_interval'] = 'NOT_ESTIMATED: cross-operator native-noise conditioning not implemented'
    data = dict(r=mean[:, 0], clean_mean=mean[:, 1:], transformed_mean=z, input=y,
        observation_mask=mask, target=generated['observations'], truth=truth,
        canonical_noise_variance=variance)
    if uncertainty is not None:
        data['clean_variance'] = uncertainty[:, 1:]
        data['r_variance'] = uncertainty[:, 0]
    if task['solver'] == 'O1':
        noisy_mean, noisy_var = gaussian_noisy_target(y, p, c, operator, variance, mask, cfg['N2']['rank_rtol'])
        widths = 1.95996398454*np.sqrt(noisy_var)
        error = generated['observations']-noisy_mean
        row['noisy_prediction_interval'] = 'conditional Gaussian approximation with shared native noise R_target,input'
        row['noisy_prediction'] = dict(coverage95=np.mean(abs(error) <= widths+1e-9, axis=0),
            mean_interval_width=np.mean(2*widths, axis=0),
            hidden_coverage95=np.mean((abs(error) <= widths+1e-9)[hidden_times], axis=0) if hidden_times.any() else None)
        data.update(noisy_prediction_mean=noisy_mean, noisy_prediction_variance=noisy_var)
    if task.get('rank_check'):
        row['rank_sensitivity'] = []
        for tolerance in cfg['N2']['rank_check_rtols']:
            try:
                if task['solver'] == 'O1':
                    other = joint.smooth_balloon_trajectory_reference(y, p, config=c, trajectory_spec=operator,
                        noise_variance=variance, input_mask=mask, rank_rtol=tolerance)
                    status = 'completed'
                else:
                    other = batch_map.smooth_balloon_trajectory_map(y, p, config=c, trajectory_spec=operator,
                        noise_variance=variance, input_mask=mask, rank_rtol=tolerance, max_nfev=cfg['budget']['map_max_nfev'])
                    status = other['status']
                row['rank_sensitivity'].append(dict(rank_rtol=tolerance, status=status, retained_rank=other['retained_rank'],
                    maximum_state_mean_difference=float(np.max(abs(other['transformed_mean']-z))) if status == 'completed' else None))
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    save_fit(run_dir, task['id'], 'fit', data, row)
                    raise
                row['rank_sensitivity'].append(dict(rank_rtol=tolerance, **failure(exc)))
    save_fit(run_dir, task['id'], 'fit', data, row)
    return dict(status='completed', rows=[row], actual_solves=3 if task.get('rank_check') else 1)


def gaussian_noisy_target(y, p, c, operator, noise, mask, rtol):
    """Condition the SAME canonical noisy target, keeping R_target,input."""
    steps = len(operator.input_time)
    prior = joint.linearized_trajectory_prior(steps, p, c)
    h = np.kron(np.eye(steps), core.observation_jacobian(np.zeros(6), p))
    covariance = h@prior@h.T+np.diag(np.tile(noise, steps))
    matrix, visible = operator.matrix(mask)
    visible &= np.isfinite(y)
    selected = matrix[visible.ravel()]
    u, singular, vt = np.linalg.svd(selected, full_matrices=False)
    keep = singular > singular[0]*rtol if len(singular) else np.zeros(0, bool)
    basis = vt[keep]
    cross = covariance@basis.T
    target = (u[:, keep].T@y[visible])/singular[keep]
    solve = np.linalg.solve(basis@cross, np.column_stack((target, cross.T))) if keep.any() else np.zeros((0, 1+3*steps))
    mean = cross@solve[:, 0]
    posterior = covariance-cross@solve[:, 1:]
    diagonal = np.diag(posterior).copy()
    if np.min(diagonal) < -1e-9:
        raise FloatingPointError('invalid conditional noisy-target covariance')
    # Remove only roundoff in a covariance whose observed coordinates have zero
    # conditional variance; no state or physical trajectory is projected.
    diagonal[(np.abs(diagonal) < 1e-12) | (diagonal < 0)] = 0.
    return mean.reshape(steps, 3), diagonal.reshape(steps, 3)


def synthetic_cell(run_dir, cfg, dc, base, task):
    condition = task['condition']
    truth_g, truth_w, truth_gain = 0., 0., 1.
    if task['family'] == 'N7' and condition in cfg['N7']['synthetic_truth_conditions']:
        index = task['replicate'] % 2
        truth_g = cfg['N7']['truth_g_values'][index] if condition in ('true_g', 'true_gw') else 0.
        truth_w = cfg['N7']['truth_w_values'][index] if condition in ('true_w', 'true_gw') else 0.
        truth_gain = cfg['N7']['truth_observation_gains'][index] if condition == 'measurement_gain' else 1.
    generated = generated_trial(base, 'nonlinear_student_t', task['seed'], cfg['synthetic']['steps'],
        truth_g=truth_g, truth_w=truth_w, truth_gain=truth_gain)
    clean = generated['clean'].copy()
    y = generated['observations'].copy()
    rng = np.random.default_rng(task['seed']+500000)
    sd = np.maximum(np.std(clean, axis=0), 1e-12)
    stress = cfg['synthetic']['stress_scale']
    if condition == 'combined':
        y = diagnostic.bridge_transform(y, 'combined', dc)
        clean = diagnostic.bridge_transform(clean, 'combined', dc)
    elif condition == 'drift':
        y += np.sin(np.linspace(0, np.pi*1.5, len(y)))[:, None]*sd*stress['drift_sd_multiplier']
    elif condition == 'correlated_hb':
        common = rng.normal(size=len(y))
        rho = stress['hb_correlation']
        innovation = np.sqrt(rho)*common[:, None]+np.sqrt(1-rho)*rng.normal(size=(len(y), 2))
        y[:, 1:] += innovation*np.asarray(base['model']['observation_scale'])[1:]
    elif condition == 'outliers':
        injected = rng.random(y.shape) < stress['outlier_fraction']
        y += injected*rng.choice([-1., 1.], y.shape)*sd*stress['outlier_sd_multiplier']
    elif condition == 'independent_pairing':
        donor = generated_trial(base, 'nonlinear_student_t', task['seed']+600000, len(y))
        y[:, 1:] = donor['observations'][:, 1:]
        clean[:, 1:] = donor['clean'][:, 1:]
    p, c, spec = model(base, task['candidate'])
    truth = np.column_stack((generated['states'][:, 0], clean))
    rows = []
    for mode in ('full', 'EEG_only', 'fNIRS_only'):
        supplied = y.copy()
        if mode != 'full':
            supplied[:, [1, 2] if mode == 'EEG_only' else [0]] = np.nan
        row = infer_row(run_dir, task, mode, supplied, y, sd, p, c, spec, mode=mode, truth=truth)
        row['reference_truth'] = dict(
            raw_noisy=clean_metrics(truth, np.column_stack((y[:, 0], y))),
            own_smoothing=clean_metrics(truth, np.column_stack((gaussian_filter1d(y[:, 0], 2.), gaussian_filter1d(y, 2., axis=0)))))
        row['shared_r_truth'] = 'EEG driver only; cross-modal relation deliberately broken' if condition == 'independent_pairing' else 'matched driver'
        if mode == 'full' and row['status'] == 'completed' and task['family'] in ('N1', 'N6'):
            try:
                with np.load(Path(run_dir)/row['trajectory_path'], allow_pickle=False) as data:
                    z = data['transformed_mean'].copy()
                    joint_mean = data['clean_mean'].copy()
                rz, physical, integration = driver_replay(z, p, c)
                replay_mean = np.array([core._observation_map_unchecked(v, p, spec) for v in physical])
                row['r_driver_replay'] = dict(status='completed', integration=integration,
                    gap_nrmse=np.sqrt(np.mean(((replay_mean-joint_mean)/sd)**2, axis=0)),
                    truth=clean_metrics(truth, np.column_stack((rz[:, 0], replay_mean))))
                save_fit(run_dir, task['id'], 'replay', dict(r=rz[:, 0], clean_mean=replay_mean,
                    target=y, truth=truth, input=y, observation_mask=np.isfinite(y), transformed_mean=rz), row['r_driver_replay'])
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    raise
                row['r_driver_replay'] = failure(exc)
        save_fit(run_dir, task['id'], mode, None, row)
        rows.append(row)
    return dict(status='completed', rows=rows, actual_solves=3, condition=condition, stream=task['stream'],
        generating_parameters=dict(g=truth_g, w=truth_w, observation_gain=truth_gain),
        truth_used_for='generator and scorer only; fixed candidate receives no generating parameters')


def curve_cell(run_dir, cfg, base, task):
    info, arrays = load_projection(str(run_dir), task['subject'], None, None, 'E0')
    identities = read_json(Path(run_dir)/'prepared'/f"{task['subject']}.json")['trials']
    p, c, spec = model(base, task['candidate'], arrays['noise'])
    rows = []
    for i in range(24):
        row = infer_row(run_dir, task, str(i), arrays['target'][i], arrays['target'][i],
                        arrays['normalizer'], p, c, spec, mode='curve')
        row.update(trial=i, sample_id=identities[i]['sample_id'], subject=task['subject'],
                   session=identities[i]['session'], w=task['candidate']['w'], gain=task['candidate']['gain'],
                   g=task['candidate'].get('g', 0.))
        rows.append(row)
    return dict(status='completed', rows=rows, actual_solves=24, coordinate_use=info['coordinate_use'])


def session_summary(run_dir, cfg, task):
    subject = task['subject']
    gw = task['family'] == 'N7'
    axis, reference_value = ('g', 0.) if gw else ('gain', 1.)
    ids = read_json(Path(run_dir)/'prepared'/f'{subject}.json')['trials']
    registered = {t['id']: t for t in read_tasks(run_dir)}
    grid = [(registered[k]['candidate']['w'], registered[k]['candidate'].get(axis, reference_value)) for k in task['curve_cells']]
    likelihood = np.full((24, len(grid)), np.nan)
    trajectory_paths = {}
    for j, key in enumerate(task['curve_cells']):
        for row in cell_result(run_dir, key).get('rows', []):
            if row['status'] == 'completed':
                likelihood[row['trial'], j] = row['parameter_log_likelihood']
                trajectory_paths[(row['trial'], j)] = row['trajectory_path']
    rows = []
    rng = np.random.default_rng(cfg['seed']+int(subject[-2:]))
    diagonal_indices = [j for j, (_, value) in enumerate(grid) if value == reference_value]
    for session in cfg['sessions']:
        indices = [i for i, identity in enumerate(ids) if identity['session'] == session]
        panel = likelihood[np.ix_(indices, diagonal_indices)]
        complete = bool(np.isfinite(panel).all())
        record = dict(subject=subject, session=session, expected_trials=8,
            completed_grid_values=int(np.isfinite(panel).sum()), expected_grid_values=panel.size,
            status='completed' if complete else 'failed_numerical',
            coordinate_use='within_subject_common_gauge_descriptive_only',
            bootstrap_scope='conditional on fixed gauge; excludes projection/noise-estimation uncertainty',
            parameter_interval='NOT_ESTIMATED: coarse likelihood grid is not a posterior interval')
        if complete:
            curve = panel.sum(axis=0)
            w = np.array([grid[j][0] for j in diagonal_indices])
            best = int(np.argmax(curve))
            boots = rng.integers(0, len(indices), (cfg['N5']['bootstrap_repetitions'], len(indices)))
            estimates = w[np.argmax(panel[boots].sum(axis=1), axis=1)]
            half = [w[np.argmax(panel[parity::2].sum(axis=0))] for parity in (0, 1)]
            near = np.flatnonzero(curve >= curve.max()-2.)
            gaps = []
            for i in indices:
                reference_path = Path(run_dir)/trajectory_paths[(i, diagonal_indices[best])]
                with np.load(reference_path, allow_pickle=False) as data:
                    mean = data['clean_mean'].copy()
                    norm = data['normalization_sd'].copy()
                for j in near:
                    with np.load(Path(run_dir)/trajectory_paths[(i, diagonal_indices[j])], allow_pickle=False) as data:
                        gaps.append(float(np.sqrt(np.mean(((data['clean_mean']-mean)/norm)**2))))
            record.update(w_grid=w, log_likelihood=curve, estimate_w=w[best],
                boundary=bool(best in (0, len(w)-1)), likelihood_range=float(np.ptp(curve)),
                near_optimal_w=w[near], near_optimal_threshold_delta_log_likelihood=2.,
                near_optimal_teacher_nrmse_max=max(gaps, default=None), half_sample_w=half,
                half_sample_absolute_difference=float(abs(half[0]-half[1])), bootstrap_w=estimates,
                bootstrap_quantiles_descriptive=np.quantile(estimates, [.025, .5, .975]),
                bootstrap_boundary_fraction=float(np.mean(np.isin(estimates, [w[0], w[-1]]))))
        surface = likelihood[indices]
        record['surface'] = dict(grid_columns=['W', 'G' if gw else 'a_N'], grid=grid,
            complete=bool(np.isfinite(surface).all()),
            completed_grid_values=int(np.isfinite(surface).sum()), expected_grid_values=surface.size,
            summed_log_likelihood=[float(surface[:, j].sum()) if np.isfinite(surface[:, j]).all() else None
                                   for j in range(len(grid))])
        if gw and record['surface']['complete']:
            scores = surface.sum(axis=0)
            best = int(np.argmax(scores))
            near = np.flatnonzero(scores >= scores.max()-2.)
            gaps, driver_gaps = [], []
            for i in indices:
                with np.load(Path(run_dir)/trajectory_paths[(i, best)], allow_pickle=False) as data:
                    mean, driver, norm = data['clean_mean'].copy(), data['r'].copy(), data['normalization_sd'].copy()
                for j in near:
                    with np.load(Path(run_dir)/trajectory_paths[(i, int(j))], allow_pickle=False) as data:
                        gaps.append(float(np.sqrt(np.mean(((data['clean_mean']-mean)/norm)**2))))
                        driver_gaps.append(float(np.sqrt(np.mean(((data['r']-driver)/norm[0])**2))))
            record['surface'].update(estimate_w=grid[best][0], estimate_g=grid[best][1],
                w_boundary=grid[best][0] in (-.5, .5), g_boundary=grid[best][1] in (-.6, .6),
                near_optimal_grid=[grid[j] for j in near], near_optimal_delta_log_likelihood=2.,
                near_optimal_teacher_nrmse_max=max(gaps, default=None),
                near_optimal_r_difference_in_eeg_training_sd_max=max(driver_gaps, default=None),
                interpretation='coarse diagnostic surface; no posterior interval or unique physiological attribution')
        if not gw:
            record['surface']['w_gain_grid'] = grid
        rows.append(record)
    complete_rows = [r for r in rows if r['status'] == 'completed']
    differences = [dict(sessions=[a['session'], b['session']], absolute_w_difference=abs(a['estimate_w']-b['estimate_w']))
                   for i, a in enumerate(complete_rows) for b in complete_rows[i+1:]]
    np.savez_compressed(Path(run_dir)/'cells'/task['id']/'trial_likelihood_grid.npz',
                        log_likelihood=likelihood, **{'w_g' if gw else 'w_gain': np.array(grid)})
    return dict(status='completed', rows=rows, session_differences=differences, actual_solves=0,
                interpretation='three-session within-subject diagnostic; not population ICC or clinical traits')


def driver_replay(z, p, c):
    """Independent five-state integration under piecewise-linear frozen r(t)."""
    clock = np.arange(len(z))*c.dt
    driver = z[:, 0]

    def rhs(t, hemo):
        state = np.r_[np.interp(t, clock, driver), hemo]
        return core.balloon_rhs(state, p, dt=c.dt)[1:]

    result = solve_ivp(rhs, (clock[0], clock[-1]), z[0, 1:], method='DOP853',
                       t_eval=clock, rtol=1e-8, atol=1e-10, max_step=c.dt/2)
    if not result.success or result.y.shape != (5, len(clock)):
        raise FloatingPointError('r-driven integration failed: '+result.message)
    transformed = np.column_stack((driver, result.y.T))
    physical = np.array([core.transformed_to_physical(row) for row in transformed])
    return transformed, physical, dict(nfev=result.nfev, method='DOP853',
        driver='piecewise linear frozen posterior-mean r; initial hemodynamic state fixed once')


def replay_cell(run_dir, cfg, base, task):
    baseline = cell_result(run_dir, task['baseline_cell'])
    full = next((r for r in baseline.get('rows', []) if r.get('mode') == 'full'), None)
    if not full or full['status'] != 'completed':
        return dict(status=full['status'] if full else 'failed_contract', rows=[], error='registered full joint fit unavailable')
    with np.load(Path(run_dir)/full['trajectory_path'], allow_pickle=False) as data:
        z, clean, y, sd = [data[k].copy() for k in ('transformed_mean', 'clean_mean', 'target', 'normalization_sd')]
    _, prepared = load_projection(str(run_dir), task['subject'], task['outer'], None, 'E0')
    p, c, spec = model(base, BASE, prepared['noise'])
    replay_z, physical, integration = driver_replay(z, p, c)
    prediction = np.array([core._observation_map_unchecked(row, p, spec) for row in physical])
    innovations, departures = [], []
    for t in range(len(z)-1):
        try:
            innovations.append((z[t+1]-core.rk4_transition(z[t], p, c))/(np.asarray(p.fixed.process_std)*np.sqrt(c.dt)))
        except core.FlowDomainExit as exc:
            departures.append(dict(time_index=t, diagnostic=exc.diagnostic))
            innovations.append(np.full(6, np.nan))
    increments = np.array(innovations)
    complete = np.isfinite(increments).all()
    row = dict(status='completed', sample_id=full['sample_id'], subject=task['subject'], session=full['session'],
        replay_gap_nrmse=np.sqrt(np.mean(((prediction-clean)/sd)**2, axis=0)),
        replay_vs_observed=residual_metrics(y, prediction, sd), joint_vs_observed=residual_metrics(y, clean, sd),
        standardized_transition_rms=(np.sqrt(np.mean(increments**2, axis=0)) if complete else None),
        grouped_transition_rms=(dict(r=float(np.sqrt(np.mean(increments[:, 0]**2))),
            hemodynamic=float(np.sqrt(np.mean(increments[:, 1:]**2)))) if complete else None),
        transition_domain_failures=departures, complete_transition_denominator=len(z)-1,
        integration=integration, physical_checks=core.run_physical_checks(physical, p),
        interpretation='posterior-mean closure diagnostic, not private-information proportion; F(E[z]) differs from E[F(z)]')
    save_fit(run_dir, task['id'], 'replay', dict(r=z[:, 0], clean_mean=prediction, transformed_mean=replay_z,
        joint_mean=clean, target=y, input=y, observation_mask=np.isfinite(y), normalization_sd=sd,
        standardized_transitions=increments), row)
    return dict(status='completed', rows=[row], actual_solves=0)


def trace_cell(run_dir, cfg, dc, base, task):
    replay_cfg = dict(previous_run=cfg['N5']['previous_run'])
    arrays, identity = repair.load_replay_inputs(replay_cfg, dc, base, task['subject'])
    old = read_json(ROOT/cfg['N5']['previous_run']/'summary.json')
    noise = old['preparation'][task['subject']]['broadband_pca_noise_scale']
    p, c, spec = model(base, dict(BASE, w=task['w']), noise)
    y = arrays['broadband_pca'][task['trial']].copy()
    y[:, 0] = np.nan
    row = dict(subject=task['subject'], training_trial_index=task['trial'], w=task['w'],
        sample_id=identity['trials'][task['trial']]['sample_id'],
        original_ma_trial_position=identity['trials'][task['trial']]['original_ma_trial_position'],
        input_identity=identity['input_sha256'], local_counterfactuals=[],
        O2_comparison='NOT_IMPLEMENTED: native feature-layer boundary absent')
    try:
        fit = joint.smooth_balloon_joint(y, p, config=c, observation_spec=spec, quadrature_order=cfg['quadrature_order'])
        row.update(status='completed', parameter_log_likelihood=fit.parameter_log_likelihood)
    except core.FlowDomainExit as exc:
        row.update(failure(exc))
        history = exc.diagnostic['update_history']
        row['first_zero_event_time_s'] = (exc.diagnostic['transition_index']-1)*c.dt-5+exc.diagnostic['first_zero_time_s']
        risky = []
        for update in history:
            before = core.flow_drift_diagnostic(update['predicted_mean'], p, c.dt)
            after = core.flow_drift_diagnostic(update['filtered_mean'], p, c.dt)
            risky.append(dict(time_index=update['time_index'], before=before, after=after))
        row['retained_update_risk_trace'] = risky
        first = next((r for r in risky if not r['after']['in_domain']), None)
        row['first_risky_update_index'] = first['time_index'] if first else None
        row['first_risky_update_event_time_s'] = first['time_index']*c.dt-5 if first else None
        row['risk_definition'] = 'updated mean whose next full dt drift exits f>0; every previous transition passed the same owning check'
        row['risk_trace_scope'] = 'last four updates retained by owning filter; does not infer earlier risk over a different horizon'
        update = history[-1]
        mean, covariance = np.array(update['predicted_mean']), np.array(update['predicted_covariance'])
        t = update['time_index']
        for label, available in [('skip_last_update', [False, False, False]),
                                 ('EEG_only_update', [True, False, False]), ('fNIRS_only_update', [False, True, True])]:
            available = np.array(available)&np.isfinite(y[t])
            changed, changed_cov, _, _ = joint.joint_observation_update(mean, covariance, y[t], available, p, spec,
                                                                       cfg['quadrature_order'], False)
            row['local_counterfactuals'].append(dict(kind=label, time_index=t, input_prefix_unchanged=True,
                observation_mask=available, drift=core.flow_drift_diagnostic(changed, p, c.dt),
                changed_mean=changed, changed_covariance=changed_cov,
                interpretation='local diagnostic; no new independent trial or deployable candidate'))
    save_fit(run_dir, task['id'], 'trace', dict(input=y, target=y, observation_mask=np.isfinite(y)), row)
    return dict(status='completed', rows=[row], actual_solves=1)


def artifact_cell(run_dir, cfg, base, task):
    subject, eeg = task['subject'], task['eeg']
    info, arrays = load_projection(str(run_dir), subject, 0, None, eeg)
    detail = read_json(Path(run_dir)/'prepared'/f'{subject}.json')
    with np.load(Path(run_dir)/'prepared'/f'{subject}.npz', allow_pickle=False) as archive:
        raw, auxiliary = archive['raw_eeg'], archive['raw_eog']
    trial = info['validation'][0]
    names = detail['eeg_channels']
    training_sd = np.std(np.concatenate(raw[info['train']]), axis=0)
    local_indices = [names.index(n) for n in info['selected_eeg_channels']]
    projection = {k: np.array(v) if isinstance(v, list) else v for k, v in info['projection'].items()}
    train_eog = auxiliary[info['train'][0]]
    if train_eog.shape[-1]:
        waveform = train_eog[:, 0]-train_eog[:, 0].mean()
        source = 'independent outer-training native EOG waveform'
    else:
        clock = np.arange(raw.shape[1])/200.
        waveform = sum(np.exp(-.5*((clock-center)/.12)**2) for center in (7., 13., 19.))
        waveform -= waveform.mean()
        source = 'fixed blink-like pulses; EOG regression not evaluable without actual references'
    waveform /= max(np.std(waveform), 1e-12)
    p, c, spec = model(base, BASE, arrays['noise'])
    rows = [infer_row(run_dir, task, 'reference', arrays['target'][trial], arrays['target'][trial], arrays['normalizer'], p, c, spec)]
    reference = rows[0]
    for spatial in cfg['N4']['artifact_spatial_weights']:
        excluded = {n.casefold() for n in cfg['N4']['excluded_frontal_channels']}
        weights = np.array([(name.casefold() in excluded if spatial == 'frontal' else name.casefold() not in excluded) for name in names])
        for strength in cfg['N4']['artifact_sd_multipliers']:
            injected = raw[trial]+waveform[:, None]*training_sd*weights*strength
            eog = auxiliary[trial].copy()
            if eeg == 'E1':
                # The injected reference is known and also enters the EOG channel;
                # train-fitted regression is applied unchanged.
                eog[:, 0] += waveform*np.std(np.concatenate(auxiliary[info['train']])[:, 0])*strength
                reg = info['regression']
                injected = cleaned_eeg(injected, eog, np.array(reg['center']), np.array(reg['coefficients']))
            feature = native_eeg_feature(injected[:, local_indices], [8., 13.] if eeg in ('E4', 'E5') else [1., 45.])
            pc = ((feature-projection['eeg_center'])/projection['eeg_feature_scale']-projection['pca_center'])@projection['loading']
            pc = (pc-pc[:20].mean())*projection['eeg_factor']*(-1 if eeg == 'E5' else 1)
            y = arrays['target'][trial].copy()
            y[:, 0] = pc
            row = infer_row(run_dir, task, task_id(spatial, strength), y, arrays['target'][trial],
                            arrays['normalizer'], p, c, spec)
            row.update(spatial=spatial, strength_training_sd=strength, contamination_source=source,
                reference_meaning='frozen un-injected measured window, not clean neural truth')
            if reference['status'] == row['status'] == 'completed':
                with np.load(Path(run_dir)/reference['trajectory_path'], allow_pickle=False) as a, \
                     np.load(Path(run_dir)/row['trajectory_path'], allow_pickle=False) as b:
                    row['r_change_rms'] = float(np.sqrt(np.mean((a['r']-b['r'])**2)))
                    row['teacher_change_nrmse'] = np.sqrt(np.mean(((a['clean_mean']-b['clean_mean'])/arrays['normalizer'])**2, axis=0))
            rows.append(row)
    return dict(status='completed', rows=rows, actual_solves=5, trial=trial,
                sample_id=detail['trials'][trial]['sample_id'], candidate=eeg)


def noise_diagnostic(run_dir, cfg, base, task):
    info, arrays = load_projection(str(run_dir), task['subject'], task['outer'], None, 'E0')
    y = arrays['target'][info['train']]
    difference = np.diff(y, axis=1)
    constant = difference_noise_constant(base['model']['student_nu'])
    rng = np.random.default_rng(cfg['seed']+int(task['subject'][-2:])*10+task['outer'])
    # Moving blocks are drawn within a trial. Trial resets never become
    # adjacent differences, and no outer-evaluation sample enters this panel.
    block_length = 8
    blocks_per_trial = int(np.ceil(difference.shape[1]/block_length))
    estimates = []
    for _ in range(200):
        samples = []
        for _ in range(len(y)):
            trial = rng.integers(len(y))
            starts = rng.integers(0, difference.shape[1]-block_length+1, blocks_per_trial)
            samples.append(np.concatenate([difference[trial, start:start+block_length] for start in starts])[:difference.shape[1]])
        estimates.append(step5.robust_mad(np.concatenate(samples))/1.482602218505602/constant)
    structures = [noise_structure(row) for row in difference]
    row = dict(status='completed', subject=task['subject'], outer=task['outer'], train=info['train'],
        first_difference_scale=step5.first_difference_noise(y, constant), effective_noise_scale=arrays['noise'],
        trial_difference_structure=structures, bootstrap_scale_quantiles=np.quantile(estimates, [.025, .5, .975], axis=0),
        bootstrap_repetitions=200, block_length_steps=8, independent_units='training trials, not time points')
    return dict(status='completed', rows=[row], actual_solves=0)


def engineering_checks(cfg, dc, base):
    """Synthetic/software checks only: safe to run before native metadata."""
    checks = {}
    p, c, spec = model(base, BASE)
    rng = np.random.default_rng(912)
    z = rng.normal(size=6)*.005
    gain_spec = replace(spec, fnirs_gain=1.5)
    physical = core.transformed_to_physical(z)
    np.testing.assert_allclose(core.observation_map(physical, p, gain_spec),
        core.observation_map(physical, p, spec)*[1., 1.5, 1.5], rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(gain_spec.effective_noise_scale, spec.effective_noise_scale)
    eps = 1e-6
    numeric = np.column_stack([(core.observation_map(core.transformed_to_physical(z+eps*direction), p, gain_spec)-
        core.observation_map(core.transformed_to_physical(z-eps*direction), p, gain_spec))/(2*eps) for direction in np.eye(6)])
    gain_error = float(np.max(abs(numeric-core.observation_jacobian(z, p, gain_spec))))
    if gain_error > 1e-8:
        raise AssertionError('mean-only gain Jacobian failed')
    checks['gain'] = dict(passed=True, derivative_max_absolute_error=gain_error, noise_unchanged=True)
    if 'N7' in cfg:
        errors = []
        reference = base['model']['reference']
        for candidate in candidates(cfg, 'N7'):
            params, numerics, observation = model(base, candidate)
            expected = [reference['beta']*np.exp(candidate['g']+2*candidate['w']),
                        reference['gamma']*np.exp(2*candidate['w']),
                        reference['kappa']*np.exp(candidate['w'])]
            actual = [params.fixed.neurovascular_gain, params.fixed.gamma, params.free.kappa]
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
            np.testing.assert_array_equal(params.fixed.process_std, p.fixed.process_std)
            np.testing.assert_array_equal(observation.effective_noise_scale, spec.effective_noise_scale)
            assert (params.free.tau, params.fixed.alpha, params.fixed.E0, params.fixed.P0, params.fixed.Q0) == (
                p.free.tau, p.fixed.alpha, p.fixed.E0, p.fixed.P0, p.fixed.Q0)
            derivative = core.rk4_transition_with_jacobian(z, params, numerics)[1]
            finite = np.column_stack([(core.rk4_transition(z+eps*d, params, numerics)-
                core.rk4_transition(z-eps*d, params, numerics))/(2*eps) for d in np.eye(6)])
            errors.append(float(np.max(abs(derivative-finite))))
        if max(errors) > 1e-7:
            raise AssertionError('G-W transition derivative mismatch')
        checks['GW'] = dict(passed=True, candidates=len(errors), derivative_max_absolute_error=max(errors),
            fixed_z_tau_alpha_E0_P0_Q0=True, fixed_observation_gain_noise_and_process=True,
            qualification='software only; no parameter or teacher qualification')
    raw = rng.normal(size=(6000, 4))
    eog = rng.normal(size=(6000, 2))
    contaminated, eog_changed = raw.copy(), eog.copy()
    contaminated[2600:3400] = 1e80
    eog_changed[2600:3400] = -1e80
    coeff = rng.normal(size=(2, 4))
    a = native_eeg_feature(cleaned_eeg(raw, eog, np.zeros(2), coeff, True), [1., 45.], True)
    b = native_eeg_feature(cleaned_eeg(contaminated, eog_changed, np.zeros(2), coeff, True), [1., 45.], True)
    np.testing.assert_array_equal(a, b)
    checks['native_hidden'] = dict(passed=True, error=0., includes='EEG and EOG before regression/filter/power')
    # Synthetic fold/identity contract does not require local campaign evidence.
    fake = [dict(subject='subject_01', session=s, training_ordinal=i, original_ma_trial_position=p,
                 event_index=p, sample_id=f'{s}/{p}') for s in cfg['sessions']
            for i, p in enumerate([0, 1, 2, 3, 5, 6, 7, 8])]
    validate_identities(fake, cfg, 'subject_01')
    checks['folds'] = dict(passed=True, outer=4, inner=3, per_session_outer=[6, 2])
    try:
        count = 12
        operator = repair.trajectory_operator(count, 'model', dc)
        generated = repair.generate_linear_gaussian(base, count, 719)
        noise = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
        reference = joint.smooth_balloon_trajectory_reference(generated['observations'], p, config=c,
            trajectory_spec=operator, noise_variance=noise)
        fit = batch_map.smooth_balloon_trajectory_map(generated['observations'], p, config=c,
            trajectory_spec=operator, noise_variance=noise, linear=True)
        if fit['status'] != 'completed':
            raise AssertionError('linear MAP did not converge')
        mean_error = float(np.linalg.norm(fit['transformed_mean']-reference['transformed_mean'])/
                           max(np.linalg.norm(reference['transformed_mean']), 1e-12))
        objective = batch_map.TrajectoryObjective(generated['observations'], p, c, operator, noise)
        path = rng.normal(size=count*6)*.001
        analytic = objective.evaluate(path)[1]
        numeric = np.column_stack([(objective.evaluate(path+eps*direction, derivative=False)[0]-
            objective.evaluate(path-eps*direction, derivative=False)[0])/(2*eps) for direction in np.eye(len(path))])
        derivative_error = float(np.linalg.norm(numeric-analytic)/max(np.linalg.norm(analytic), 1e-12))
        # A known unit change adjusts density volume, while leaving the MAP.
        scaled_spec = spec.reexpress([1.3, .7, 1.8])
        changed = batch_map.TrajectoryObjective(generated['observations']*scaled_spec.coordinate_scale,
            p, c, operator, noise, observation_spec=scaled_spec)
        unscaled_residual = objective.evaluate(path)[0]
        scaled_residual = changed.evaluate(path)[0]
        density_error = float(abs((changed.observation_normalization+scaled_residual@scaled_residual/2)-
            (objective.observation_normalization+unscaled_residual@unscaled_residual/2)-
            scaled_spec.log_abs_det(np.ones_like(generated['observations'], dtype=bool))))
        mask = repair.bridge_mask(32, 'center_EEG', 16)
        masked_operator = repair.trajectory_operator(32, 'combined', dc).with_visible_interpolation(mask, mask)
        input_data = rng.normal(size=(32, 3))*.01
        replaced = input_data.copy()
        replaced[~mask] = 1e90
        ya, yb = masked_operator.apply(input_data, mask), masked_operator.apply(replaced, mask)
        np.testing.assert_array_equal(ya, yb)
        before = batch_map.TrajectoryObjective(ya, p, c, masked_operator, noise, input_mask=mask)
        after = batch_map.TrajectoryObjective(yb, p, c, masked_operator, noise, input_mask=mask)
        hidden_error = float(np.max(abs(before.evaluate(np.zeros(32*6))[0]-after.evaluate(np.zeros(32*6))[0])))
        measured = dict(linear_mean=mean_error, derivative=derivative_error,
                        hidden_intervention=hidden_error, density_units=density_error)
        passed = all(measured[k] <= cfg['N2']['engineering_tolerances'][k] for k in measured)
        checks['O2'] = dict(passed=passed, measured=measured, tolerances=cfg['N2']['engineering_tolerances'],
            variance='NOT_ESTIMATED', measured_branch='NOT_IMPLEMENTED: nonlinear native preprocessing boundary not exposed')
    except Exception as exc:
        checks['O2'] = dict(passed=False, error=repr(exc), traceback=traceback.format_exc())
    return checks


def execute_cell(run_dir, cfg, task):
    """Worker entry. A cell alarm interrupts only this worker, not its siblings."""
    started = time.monotonic()
    directory = Path(run_dir)/'cells'/task['id']
    directory.mkdir(parents=True, exist_ok=True)
    def timeout_handler(signum, frame):
        raise TimeoutError('registered cell time budget exhausted')
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, task.get('timeout_seconds', cfg['budget']['cell_timeout_seconds']))
    try:
        _, dc, base, measured, metadata = load_config(Path(run_dir)/'resolved_config.yaml')
        kind = task['kind']
        if kind in ('inner', 'outer'):
            result = measured_cell(run_dir, cfg, base, task)
        elif kind == 'select':
            result = select_candidate(run_dir, cfg, task)
        elif kind == 'selected_outer':
            result = selected_outer(run_dir, cfg, base, task)
        elif kind == 'temporal':
            result = temporal_cell(run_dir, cfg, base, task)
        elif kind == 'synthetic':
            result = synthetic_cell(run_dir, cfg, dc, base, task)
        elif kind == 'linear':
            result = linear_cell(run_dir, cfg, dc, task)
        elif kind == 'curve':
            result = curve_cell(run_dir, cfg, base, task)
        elif kind == 'session_summary':
            result = session_summary(run_dir, cfg, task)
        elif kind == 'replay':
            result = replay_cell(run_dir, cfg, base, task)
        elif kind == 'failure_trace':
            result = trace_cell(run_dir, cfg, dc, base, task)
        elif kind == 'artifact':
            result = artifact_cell(run_dir, cfg, base, task)
        elif kind == 'noise_diagnostic':
            result = noise_diagnostic(run_dir, cfg, base, task)
        elif kind == 'measured_map':
            result = dict(status='not_implemented', reason=task['reason'], rows=[])
        elif kind == 'prepare_subject':
            result = dict(status='completed', rows=[], **prepare_subject(run_dir, cfg, dc, base, measured, metadata,
                                                                        task['subject'], task['identities']))
        elif kind == 'prepare_projection':
            rows = []
            for inner in (None, 0, 1, 2) if task['outer'] is not None else (None,):
                for eeg in cfg['N4']['candidates'] if task['outer'] is not None else ('E0',):
                    try:
                        prepare_projection(run_dir, cfg, base, measured, task['subject'], task['outer'], inner, eeg)
                        rows.append(dict(inner=inner, eeg=eeg, status='completed'))
                    except Exception as exc:
                        if isinstance(exc, TimeoutError):
                            raise
                        rows.append(dict(inner=inner, eeg=eeg, **failure(exc)))
            result = dict(status='completed', rows=rows)
        else:
            raise ValueError('unknown registered cell kind')
    except Exception as exc:
        result = dict(**failure(exc), rows=partial_rows(run_dir, task))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result.update(task_id=task['id'], family=task['family'], kind=task['kind'],
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(directory/'result.json', result)
    return {k: result.get(k) for k in ('task_id', 'family', 'kind', 'status', 'elapsed_seconds', 'peak_rss_bytes', 'completed_at')}


def partial_rows(run_dir, task):
    """Recover only predeclared row names, including failures; never glob successes."""
    kind = task['kind']
    names = []
    if kind in ('outer', 'selected_outer'):
        names = [task_id(task['trial'], mode) for mode in OUTER_MASKS]
    elif kind == 'inner':
        inventory = read_json(Path(run_dir)/'scope_inventory.json')[task['subject']]
        val = folds(inventory, task['outer'])['inner'][task['inner']]['validation']
        names = [task_id(i, mode) for i in val for mode in ('center_EEG', 'center_fNIRS')]
    elif kind == 'curve':
        names = [str(i) for i in range(24)]
    elif kind == 'temporal':
        names = ['fit']
    elif kind == 'synthetic':
        names = ['full', 'EEG_only', 'fNIRS_only']
    elif kind in ('replay', 'failure_trace'):
        names = ['replay' if kind == 'replay' else 'trace']
    elif kind == 'artifact':
        names = ['reference']+[task_id(spatial, strength) for spatial in ('frontal', 'posterior') for strength in (.5, 1.)]
    paths = [Path(run_dir)/'cells'/task['id']/(name+'.json') for name in names]
    return [read_json(path) for path in paths if path.exists()]


def append_status(run_dir, result):
    with (Path(run_dir)/'case_status.jsonl').open('a') as stream:
        stream.write(json.dumps(serial(result), ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def available_memory():
    fields = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemAvailable:', 'MemTotal:'))}
    available = fields['MemAvailable']
    # Respect a container/cgroup limit as well as host MemAvailable when present.
    for base in (Path('/sys/fs/cgroup'),):
        if (base/'memory.max').exists():
            limit = (base/'memory.max').read_text().strip()
            if limit != 'max':
                used = int((base/'memory.current').read_text())
                available = min(available, max(0, int(limit)-used))
    return available


def resource_budget(cfg, pilots):
    regular = max((r['peak_rss_bytes'] for r in pilots if r.get('family') != 'N2'), default=512*1024**2)
    nonlinear = max((r['peak_rss_bytes'] for r in pilots if r.get('family') == 'N2'), default=regular)
    memory = int(available_memory()*cfg['budget']['memory_fraction'])
    cpus = len(os.sched_getaffinity(0))
    workers = min(cfg['budget']['max_workers'], max(1, cpus-2), memory//regular)
    return dict(workers=workers, available_cpus=cpus, memory_tokens_bytes=memory,
                regular_job_bytes=regular, map_job_bytes=nonlinear,
                status='ready' if workers >= 1 and memory >= min(regular, nonlinear) else 'insufficient_memory')


def freeze_sources(run_dir, cfg):
    destination = Path(run_dir)/'source_snapshot'
    if destination.exists():
        raise FileExistsError('source snapshot already frozen')
    # Git's explicit source inventory avoids scanning data, runs, cache or archive.
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    extra = ['experiments/evaluate_ssm_overnight_diagnostics.py',
             'src/inference/balloon_trajectory_map.py',
             'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v1.yaml',
             'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml']
    paths = sorted(set(tracked+extra))
    hashes = {}
    for rel in paths:
        if not rel or not (ROOT/rel).is_file():
            continue
        parts = Path(rel).parts
        if any(p in ('data', 'runs', 'cache', 'checkpoints', 'upstream', 'archive', '.git', '.venv') for p in parts):
            # src/data is source code, not a measured-data directory.
            if not (parts[0] == 'src' and parts[1] == 'data'):
                continue
        if not (rel.startswith(('src/', 'experiments/', 'research_state/')) or rel in ('ssm_next.md', 'requirements.txt', 'pytest.ini')):
            continue
        if Path(rel).suffix not in ('.py', '.yaml', '.yml', '.json', '.md', '.txt', '.ini'):
            continue
        target = destination/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (ROOT/rel).read_bytes()
        target.write_bytes(payload)
        hashes[rel] = hashlib.sha256(payload).hexdigest()
    atomic_json(Path(run_dir)/'source_snapshot_identity.json', dict(
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=hashes, project_root=ROOT, configuration='resolved_config.yaml',
        captured_working_tree=True, runtime_source='source_snapshot, not live working tree'))


def verify_snapshot(run_dir):
    identity = read_json(Path(run_dir)/'source_snapshot_identity.json')
    if CODE_ROOT != (Path(run_dir)/'source_snapshot').resolve():
        raise ValueError('run must use the frozen source_snapshot entrypoint')
    for relative, digest in identity['source_sha256'].items():
        if hashlib.sha256((CODE_ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen scientific source changed: '+relative)


def equal_subject_mean(rows, key):
    """Trial -> session -> subject, with no pooling of time points."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get(key) is not None:
            groups[row['subject']][row['session']].append(np.asarray(row[key], dtype=float))
    subject_values = {subject: np.mean([np.mean(values, axis=0) for values in sessions.values()], axis=0)
                      for subject, sessions in groups.items()}
    mean = np.mean(list(subject_values.values()), axis=0) if subject_values else None
    return mean, subject_values


def score_outer_task(task, result):
    by_mode = {r.get('mode'): r for r in result.get('rows', [])}
    if not all(mode in by_mode and by_mode[mode]['status'] == 'completed' for mode in OUTER_MASKS):
        return None
    eeg = by_mode['center_EEG']['center_metrics']['EEG']['nmse']
    hb = [by_mode['center_fNIRS']['center_metrics'][m]['nmse'] for m in ('HbO', 'HbR')]
    nmse = np.array([eeg, *hb])
    if not np.isfinite(nmse).all():
        return None
    own = np.array([by_mode['center_EEG_own']['center_metrics']['EEG']['nmse'],
                   *[by_mode['center_fNIRS_own']['center_metrics'][m]['nmse'] for m in ('HbO', 'HbR')]])
    increments = {}
    for null in ('own', 'template', 'pairing', 'shift'):
        score = np.array([by_mode['center_EEG_'+null]['center_metrics']['EEG']['nmse'],
            *[by_mode['center_fNIRS_'+null]['center_metrics'][m]['nmse'] for m in ('HbO', 'HbR')]])
        increments[null] = score-nmse
    return dict(task_id=task['id'], subject=task['subject'], session=by_mode['full']['session'],
        sample_id=by_mode['full']['sample_id'], trial=task['trial'], nmse=nmse,
        nrmse=np.sqrt(nmse), B=float(nmse@[.5, .25, .25]), fnirs=float(nmse[1:].mean()),
        increments=increments, own_nmse=own,
        selected_candidate=result.get('selected_candidate', task.get('candidate', BASE)))


def synthetic_screen(cfg, tasks, results, family, selections):
    if family == 'N4':
        rows = [r for t in tasks if t['family'] == 'N4' and t['kind'] == 'artifact'
                for r in results[t['id']].get('rows', [])]
        expected = len(cfg['subjects'])*len(cfg['N4']['candidates'])*5
        return dict(status='spatial_generator_not_available', native_artifact_completed=sum(r['status'] == 'completed' for r in rows),
                    native_artifact_expected=expected, passed=False,
                    reason='native robustness is reported separately; no clean spatial r truth available for N4')
    selected_ids = Counter(c['id'] for c in selections)
    candidate_rows, baseline_rows = {}, {}
    for task in tasks:
        if task['kind'] != 'synthetic' or task['stream'] != 'assessment' or task['condition'] != 'matched_student_t':
            continue
        fit = next((r for r in results[task['id']].get('rows', []) if r.get('mode') == 'full'), None)
        if not fit or fit['status'] != 'completed':
            continue
        if task['family'] == 'N1' and task['candidate']['id'] == 'baseline':
            baseline_rows[task['replicate']] = fit
        elif task['family'] == family:
            candidate_rows[(task['candidate']['id'], task['replicate'])] = fit
    deltas = []
    expected = cfg['synthetic']['replicates_per_condition']*sum(selected_ids.values())
    for name, frequency in selected_ids.items():
        for replicate in range(cfg['synthetic']['replicates_per_condition']):
            reference = baseline_rows.get(replicate)
            row = reference if name == 'baseline' else candidate_rows.get((name, replicate))
            if reference and row:
                delta = np.array([row['truth'][key]['nrmse']-reference['truth'][key]['nrmse']
                    for key in ('r', 'clean_EEG', 'clean_HbO', 'clean_HbR')])
                deltas.extend([delta]*frequency)
    complete = len(deltas) == expected and expected > 0
    mean = np.mean(deltas, axis=0) if deltas else None
    return dict(status='completed' if complete else 'incomplete', expected=expected, completed=len(deltas),
        mean_nrmse_change=mean, passed=bool(complete and np.all(mean <= cfg['selection']['maximum_synthetic_nrmse_degradation'])),
        scope='independent assessment matched law; weighted by inner-selected rule frequency; stress reported separately')


def candidate_summary(cfg, tasks, results):
    baseline = [score_outer_task(t, results[t['id']]) for t in tasks if t['family'] == 'N1' and t['kind'] == 'outer' and t['candidate']['w'] == 0]
    baseline = [r for r in baseline if r]
    rows = []
    rules = [(family, 'single_factor') for family in ('N3', 'N4', 'N5', 'N6')]
    if 'N7' in cfg:
        rules += [('N7', rule) for rule in cfg['N7']['rules']]
    for family, rule in rules:
        reference_rows, comparison_reference = baseline, 'N1/fixed_W=0'
        if family == 'N7' and rule == 'GW':
            reference_rows = [score_outer_task(t, results[t['id']]) for t in tasks
                if t['family'] == 'N7' and t['kind'] == 'selected_outer' and t['rule'] == 'W_only']
            reference_rows = [r for r in reference_rows if r]
            comparison_reference = 'N7/W_only'
        baseline_by_id = {r['sample_id']: r for r in reference_rows}
        selected_tasks = [t for t in tasks if t['family'] == family and t['kind'] == 'selected_outer'
                          and t.get('rule', 'single_factor') == rule]
        selected = [score_outer_task(t, results[t['id']]) for t in selected_tasks]
        complete = [r for r in selected if r]
        paired = [r for r in complete if r['sample_id'] in baseline_by_id]
        selection_cells = [results[t['id']] for t in tasks if t['family'] == family and t['kind'] == 'select'
                           and t.get('rule', 'single_factor') == rule]
        choices = [r['selected'] for r in selection_cells if r['status'] == 'completed']
        truth = synthetic_screen(cfg, tasks, results, family, choices)
        endpoint = 'fnirs' if family == 'N4' else 'B'
        value, by_subject = equal_subject_mean(paired, endpoint)
        ref_value, ref_by_subject = equal_subject_mean([baseline_by_id[r['sample_id']] for r in paired], endpoint)
        nmse, _ = equal_subject_mean(paired, 'nmse')
        base_nmse, _ = equal_subject_mean([baseline_by_id[r['sample_id']] for r in paired], 'nmse')
        risk_improvement = 1-float(value/ref_value) if value is not None and ref_value > 0 else None
        directions = sum(float(v) < float(ref_by_subject[s]) for s, v in by_subject.items())
        null_changes = {}
        for null in ('own', 'template', 'pairing', 'shift'):
            changes = [r['increments'][null]-baseline_by_id[r['sample_id']]['increments'][null] for r in paired]
            null_changes[null] = np.mean(changes, axis=0) if changes else None
        columns = slice(1, 3) if family == 'N4' else slice(0, 3)
        degradation = np.sqrt(nmse)-np.sqrt(base_nmse) if nmse is not None else None
        all_complete = len(paired) == 72 and len(choices) == 12
        mean_checks = bool(all_complete and risk_improvement >= cfg['selection']['minimum_relative_risk_improvement'] and
            np.max(degradation[columns]) <= cfg['selection']['maximum_modality_nrmse_degradation'] and
            directions >= cfg['selection']['minimum_subjects_improving'] and
            all(np.all(v[columns] >= -cfg['selection']['maximum_null_increment_degradation']) for v in null_changes.values()))
        rows.append(dict(family=family, rule=rule, comparison_reference=comparison_reference,
            status='complete_outer_rule' if all_complete else 'incomplete_outer_rule',
            expected_outer_trials=72, completed_outer_trials=len(complete), baseline_completed_outer_trials=len(reference_rows),
            common_success_trials=len(paired), expected_selection_folds=12, completed_selection_folds=len(choices),
            selected_frequency=dict(Counter(c['id'] for c in choices)), endpoint=endpoint,
            risk=float(value) if all_complete else None, baseline_risk=float(ref_value) if all_complete else None,
            relative_risk_improvement=risk_improvement if all_complete else None,
            common_success_only_risk=None if value is None else float(value),
            common_success_only_baseline_risk=None if ref_value is None else float(ref_value),
            modality_nrmse=np.sqrt(nmse) if all_complete else None,
            baseline_modality_nrmse=np.sqrt(base_nmse) if all_complete else None,
            modality_nrmse_degradation=degradation if all_complete else None,
            subjects_improving=directions, null_increment_changes=null_changes,
            synthetic_assessment=truth, measured_screen_passed=mean_checks,
            eligible_for_priority_review=bool(mean_checks and truth['passed']),
            qualification='none; selection rule evaluated once per outer fold'))
    return rows


def compromise_rows(run_dir, tasks, results):
    rows = []
    for task in tasks:
        if task['family'] != 'N1' or task['kind'] != 'outer':
            continue
        modes = {r.get('mode'): r for r in results[task['id']].get('rows', [])}
        full = modes.get('full')
        if not full or full['status'] != 'completed':
            continue
        for own, columns in [('EEG_only', ['EEG']), ('fNIRS_only', ['HbO', 'HbR'])]:
            single = modes.get(own)
            row = dict(subject=task['subject'], session=full['session'], trial=task['trial'],
                       sample_id=full['sample_id'], w=task['candidate']['w'], own=own,
                       status=single['status'] if single else 'not_started_budget')
            if single and single['status'] == 'completed':
                row['signed_full_fit_cost'] = {m: full['metrics'][m]['nmse']-single['metrics'][m]['nmse'] for m in columns}
                with np.load(Path(run_dir)/full['trajectory_path'], allow_pickle=False) as a, \
                     np.load(Path(run_dir)/single['trajectory_path'], allow_pickle=False) as b:
                    row['r_standardized_rms_difference'] = float(np.sqrt(np.mean((a['r']-b['r'])**2))/a['normalization_sd'][0])
                    row['r_correlation'] = correlation(a['r'], b['r'])
                    row['r_amplitude_ratio'] = float(np.std(a['r'])/np.std(b['r'])) if np.std(b['r']) > 1e-15 else None
                mask = 'center_EEG' if own == 'EEG_only' else 'center_fNIRS'
                a, b = modes.get(mask), modes.get(mask+'_own')
                row['signed_masked_cost'] = ({m: a['center_metrics'][m]['nmse']-b['center_metrics'][m]['nmse'] for m in columns}
                    if a and b and a['status'] == b['status'] == 'completed' else None)
            rows.append(row)
    return rows


def render_examples(run_dir, tasks, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = set()
    for task in tasks:
        if task['family'] != 'N1' or task['kind'] != 'outer' or task['candidate']['w'] != 0:
            continue
        row = next((r for r in results[task['id']].get('rows', []) if r.get('mode') == 'full' and r['status'] == 'completed'), None)
        if row is None or (task['subject'], row['session']) in selected:
            continue
        selected.add((task['subject'], row['session']))
        with np.load(Path(run_dir)/row['trajectory_path'], allow_pickle=False) as data:
            clock = np.arange(len(data['r']))/4-5
            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(clock, data['r'], color='#475c7a')
            axes[0].set_ylabel('r')
            for i, name in enumerate(MODALITIES):
                axes[i+1].plot(clock, data['target'][:, i], color='#555555', label='observed coordinate')
                axes[i+1].plot(clock, data['clean_mean'][:, i], color='#b44e32', label='joint mean')
                width = 1.95996398454*np.sqrt(data['clean_variance'][:, i])
                axes[i+1].fill_between(clock, data['clean_mean'][:, i]-width, data['clean_mean'][:, i]+width,
                                      color='#b44e32', alpha=.15, label='approximate clean interval')
                axes[i+1].set_ylabel(name)
            axes[1].legend(loc='upper right', fontsize=8)
            axes[-1].set_xlabel('event-relative time (s)')
            fig.suptitle(f"{task['subject']} / {row['session']} / training trial {task['trial']} / W=0")
            fig.tight_layout()
            directory = Path(run_dir)/'N1'
            directory.mkdir(exist_ok=True)
            fig.savefig(directory/f"{task['subject']}_{row['session']}.png", dpi=130)
            plt.close(fig)


def temporal_summary(tasks, results):
    groups = defaultdict(list)
    for task in tasks:
        if task['kind'] == 'temporal':
            groups[(task['law'], task['variant'], task['mode'], task['w'], task['solver'])].append(task)
    output = []
    for key, panel in groups.items():
        complete = [results[t['id']]['rows'][0] for t in panel if results[t['id']]['status'] == 'completed' and
                    results[t['id']].get('rows') and results[t['id']]['rows'][0]['status'] == 'completed']
        metrics = {}
        for name in ('r', 'clean_EEG', 'clean_HbO', 'clean_HbR'):
            metrics[name] = {metric: float(np.mean([r['truth'][name][metric] for r in complete
                if r['truth'][name].get(metric) is not None])) if any(r['truth'][name].get(metric) is not None for r in complete) else None
                for metric in ('nrmse', 'correlation', 'coverage95', 'mean_interval_width')}
        output.append(dict(law=key[0], variant=key[1], mask=key[2], w=key[3], solver=key[4],
            expected_independent_trials=len(panel), completed_independent_trials=len(complete),
            complete_identity_set=len(complete) == len(panel), metrics=metrics,
            interpretation='mean over completed cases with explicit full denominator; no successful-subset qualification'))
    return output


def influence_rows(run_dir, tasks, results):
    output = []
    for task in tasks:
        if task['kind'] != 'outer' or task['family'] != 'N1':
            continue
        modes = {r.get('mode'): r for r in results[task['id']].get('rows', [])}
        for target in ('EEG', 'fNIRS'):
            joint_row = modes.get('center_'+target)
            for control in ('own', 'template', 'pairing', 'shift'):
                changed = modes.get('center_'+target+'_'+control)
                row = dict(subject=task['subject'], trial=task['trial'], w=task['candidate']['w'], target=target, control=control,
                           status='completed' if joint_row and changed and joint_row['status'] == changed['status'] == 'completed' else 'incomplete')
                if row['status'] == 'completed':
                    with np.load(Path(run_dir)/joint_row['trajectory_path'], allow_pickle=False) as a, \
                         np.load(Path(run_dir)/changed['trajectory_path'], allow_pickle=False) as b:
                        row['r_change_normalized_rms'] = float(np.sqrt(np.mean((a['r']-b['r'])**2))/a['normalization_sd'][0])
                        row['r_correlation'] = correlation(a['r'], b['r'])
                    cols = ['EEG'] if target == 'EEG' else ['HbO', 'HbR']
                    row['prediction_nmse_increment_of_joint'] = {m: changed['center_metrics'][m]['nmse']-joint_row['center_metrics'][m]['nmse'] for m in cols}
                    row['session'], row['sample_id'] = joint_row['session'], joint_row['sample_id']
                output.append(row)
    return output


def summarize_run(run_dir, cfg, *, final=False):
    tasks = read_tasks(run_dir)
    results = {t['id']: cell_result(run_dir, t['id']) for t in tasks}
    family_summary, failures, scores = {}, [], defaultdict(list)
    families = FAMILIES if 'N7' in cfg else FAMILIES[:-1]
    for family in families:
        panel = [t for t in tasks if t['family'] == family]
        states = Counter(results[t['id']]['status'] for t in panel)
        planned = sum(t['planned_solves'] for t in panel)
        completed_rows = 0
        for task in panel:
            result = results[task['id']]
            completed_rows += sum(r.get('status') == 'completed' for r in result.get('rows', []) if 'trajectory_path' in r)
            if result['status'] != 'completed':
                failures.append(dict(family=family, task_id=task['id'], row_id='', status=result['status'],
                    expected_fits=task['planned_solves'], error=result.get('error', result.get('reason')),
                    identity={k: task.get(k) for k in ('subject', 'trial', 'replicate', 'seed', 'law', 'condition')}))
            for row in result.get('rows', []):
                scores[family].append({'task_id': task['id'], 'kind': task['kind'], 'rule': task.get('rule'),
                    'candidate': task.get('candidate', result.get('selected_candidate')), **row})
                if row.get('status') != 'completed':
                    flow = row.get('flow_domain_exit', {})
                    failures.append(dict(family=family, task_id=task['id'], row_id=row.get('row_id'),
                        status=row.get('status'), expected_fits=1, error=row.get('error'),
                        sample_id=row.get('sample_id'), first_transition=flow.get('transition_index'),
                        first_zero_time_s=flow.get('first_zero_time_s')))
        family_summary[family] = dict(expected_cells=len(panel), status_counts=dict(states),
            planned_model_solves=planned, actual_model_solves=sum(results[t['id']].get('actual_solves', 0) for t in panel),
            completed_trajectory_rows=completed_rows,
            expected_fit_denominator_note='planned_model_solves includes selected baseline reuse; full identities retained in task_table')
        directory = Path(run_dir)/family
        directory.mkdir(exist_ok=True)
        write_csv(directory/'trial_metrics.csv', scores[family])
        atomic_json(directory/'summary.json', family_summary[family])
    candidates_table = candidate_summary(cfg, tasks, results)
    write_csv(Path(run_dir)/'candidate_table.csv', candidates_table)
    compromises = compromise_rows(run_dir, tasks, results)
    write_csv(Path(run_dir)/'N1'/'compromise_by_subject_session.csv', compromises)
    write_csv(Path(run_dir)/'N1'/'full_fit_residuals.csv', [r for r in scores['N1'] if r.get('mode') == 'full'])
    write_csv(Path(run_dir)/'N1'/'masked_control_scores.csv', [r for r in scores['N1'] if str(r.get('mode', '')).startswith('center_')])
    write_csv(Path(run_dir)/'N1'/'innovation_structure.csv', [dict(task_id=r['task_id'],
        standardized_innovations=r.get('innovation_structure'), smoother_residuals=r.get('smoothing_residual_structure'))
        for r in scores['N1'] if r.get('mode') == 'full'])
    write_csv(Path(run_dir)/'N1'/'modality_influence.csv', influence_rows(run_dir, tasks, results))
    temporal = temporal_summary(tasks, results)
    write_csv(Path(run_dir)/'N2'/'solver_comparison.csv', temporal)
    family_summary['N2']['solver_comparison'] = temporal
    for row in candidates_table:
        if row['family'] == 'N7':
            family_summary['N7'].setdefault('selected_rules', {})[row['rule']] = row
        else:
            family_summary[row['family']]['selected_rule'] = row
    sessions = [r for t in tasks if t['family'] == 'N5' and t['kind'] == 'session_summary'
                for r in results[t['id']].get('rows', [])]
    family_summary['N5']['common_coordinate_sessions'] = sessions
    write_csv(Path(run_dir)/'N5'/'session_reproducibility.csv', sessions)
    replay = [r for t in tasks if t['kind'] == 'replay' for r in results[t['id']].get('rows', [])]
    mean_gap, _ = equal_subject_mean(replay, 'replay_gap_nrmse')
    family_summary['N6']['r_replay'] = dict(expected=72, completed=sum(r['status'] == 'completed' for r in replay),
        mean_gap_nrmse=mean_gap, interpretation='conditional completed subset; match against the synthetic replay statistics')
    if 'N7' in cfg:
        gw_sessions = [r for t in tasks if t['family'] == 'N7' and t['kind'] == 'session_summary'
                       for r in results[t['id']].get('rows', [])]
        family_summary['N7']['common_coordinate_gw_surfaces'] = gw_sessions
        write_csv(Path(run_dir)/'N7'/'gw_session_surfaces.csv', gw_sessions)
        write_csv(Path(run_dir)/'N7'/'rule_comparison.csv', [r for r in candidates_table if r['family'] in ('N5', 'N7')])
        truth_rows = []
        for task in tasks:
            if task['family'] != 'N7' or task['kind'] != 'synthetic':
                continue
            result = results[task['id']]
            full = next((r for r in result.get('rows', []) if r.get('mode') == 'full'), {})
            truth_rows.append(dict(task_id=task['id'], stream=task['stream'], condition=task['condition'],
                replicate=task['replicate'], seed=task['seed'], candidate=task['candidate'],
                status=full.get('status', result['status']), generating_parameters=result.get('generating_parameters'),
                clean_truth_metrics=full.get('truth'), likelihood=full.get('parameter_log_likelihood'),
                interpretation='fixed-candidate response surface; no truth-assisted fitting or parameter interval'))
        write_csv(Path(run_dir)/'N7'/'synthetic_response_surfaces.csv', truth_rows)
    for family in families:
        atomic_json(Path(run_dir)/family/'summary.json', family_summary[family])
    write_csv(Path(run_dir)/'failure_attribution.csv', failures)
    directions = []
    candidate_map = {r['family']: r for r in candidates_table if r['rule'] in ('single_factor', 'GW')}
    hypotheses = dict(N1='quantify modality compromise and compare own/template/pairing controls',
        N2='separate temporal observation operator from nonlinear inference', N3='relative modality confidence',
        N4='native EEG artifact/space/band/sign definition', N5='observation amplitude and W compensation',
        N6='downstream process innovations and flow-domain update origin',
        N7='independent G-W physiological gain/time adaptation versus W-only, G-only and measurement gain')
    for family in families:
        candidate = candidate_map.get(family)
        directions.append(dict(family=family, question=hypotheses[family],
            decision='priority_review' if candidate and candidate['eligible_for_priority_review'] else
                     'diagnostic_only_or_incomplete', evidence=family_summary[family],
            candidate_status=candidate['status'] if candidate else 'mechanism_diagnostic',
            cannot_claim='teacher qualification, clean measured neural truth, population ICC, or tokenizer promotion'))
    write_csv(Path(run_dir)/'direction_decision_table.csv', directions)
    n2_panel = [t for t in tasks if t['kind'] == 'temporal' and t['solver'] == 'O2' and
                t['law'] == 'nonlinear_gaussian' and t['mode'] == 'full' and t['variant'] == 'model' and t['w'] == 0]
    n2_rows = [results[t['id']].get('rows', [])[0] for t in n2_panel if results[t['id']].get('rows') and
               results[t['id']]['rows'][0]['status'] == 'completed']
    mean_truth = {key: float(np.mean([r['truth'][key]['nrmse'] for r in n2_rows])) if n2_rows else None
                  for key in ('r', 'clean_EEG', 'clean_HbO', 'clean_HbR')}
    correlations = [r['truth']['r']['correlation'] for r in n2_rows if r['truth']['r']['correlation'] is not None]
    r_correlation = float(np.mean(correlations)) if correlations else None
    n2_gate = dict(expected=24, completed=len(n2_rows), mean_nrmse=mean_truth, mean_r_correlation=r_correlation,
        mean_precheck_passed=bool(len(n2_rows) == 24 and all(v <= cfg['N2']['mean_precheck']['maximum_nrmse'] for v in mean_truth.values())
                                 and len(correlations) == 24 and r_correlation >= cfg['N2']['mean_precheck']['minimum_r_correlation']),
        measured_status='NOT_IMPLEMENTED: current native helper has no declared pre-linear feature boundary')
    atomic_json(Path(run_dir)/'N2'/'mean_precheck.json', n2_gate)
    manifest = read_json(Path(run_dir)/'manifest.json')
    lines = ['# SSM overnight diagnostics', '',
        f"Run: `{Path(run_dir).name}`. Execution: **{manifest['execution']}**. Teacher qualification: **none**.", '',
        '72 unique original training trials: subjects 01/09/18, sessions 01/03/05; original MA positions 4/9 excluded before preprocessing.',
        'Every outer fold refits all learned objects using 6 training / 2 evaluation trials per session. Three inner folds only use outer training.',
        'The N5/N7 all-training common gauge is descriptive and never enters outer candidate scoring.', '',
        '| Family | Completed cells / fixed cells | Failed/unavailable/budget cells | Actual / planned solves |',
        '|---|---:|---:|---:|']
    for family, value in family_summary.items():
        completed = value['status_counts'].get('completed', 0)
        lines.append(f"| {family} | {completed}/{value['expected_cells']} | {value['expected_cells']-completed} | {value['actual_model_solves']}/{value['planned_model_solves']} |")
    lines += ['', 'A completed cell can contain failed individual fits. See `failure_attribution.csv` and per-trial tables; no successful-only denominator is used for candidate admission.', '',
        '| Rule | Complete outer trials / 72 | Reference | Endpoint | Relative improvement | Priority review |',
        '|---|---:|---|---|---:|---|']
    for row in candidates_table:
        improvement = 'NOT_ESTIMATED' if row['relative_risk_improvement'] is None else f"{row['relative_risk_improvement']:.1%}"
        label = row['family']+('/'+row['rule'] if row['family'] == 'N7' else '')
        lines.append(f"| {label} | {row['completed_outer_trials']}/72 | {row['comparison_reference']} | {row['endpoint']} | {improvement} | {row['eligible_for_priority_review']} |")
    lines += ['', f"O2 nonlinear-Gaussian W=0/full mean precheck: {n2_gate['completed']}/24 complete; mean NRMSE {mean_truth}; r correlation {r_correlation}.",
        'O2 native measured branch: NOT_IMPLEMENTED. The current helper combines nonlinear EEG power/optics/motion transforms without an exposed pre-linear noise layer.',
        'MAP intervals and marginal likelihood: NOT_ESTIMATED. No pointwise variance is borrowed. Shared native-noise cross-operator prediction intervals are NOT_ESTIMATED.',
        'EEG voltage reconstruction: NOT_SUPPORTED after log-power/PCA. HbO/HbR retain order, shared scale and HbT algebra; no sign flip was selected.',
        'N4 local coordinates use the nearest up to three EEG locations to the fold-fixed fNIRS pair in compatible native montage metadata. If unavailable, F3 is explicitly labelled anatomical_mapping_unverified. EOG is train-fitted; artificial contamination uses an un-injected measured reference, not clean truth.',
        'N5 bootstrap: 200 trial resamples conditional on one fixed within-subject gauge; not population ICC or parameter posterior coverage.',
        'N6 mean-driver replay is not a private-information proportion. Nonlinear posterior means need not obey deterministic closure.', '',
        'N7 fits independent G/W coordinates on a fixed grid with Z and all other physiological/noise settings fixed. W-only and G-only rules reuse the same inner fits; N5 supplies the separate measurement-gain control. G and measurement gain never vary together. Extra synthetic truth shifts are response diagnostics, not SBC.',
        'Detailed evidence: `task_table.csv`, `case_status.jsonl`, `candidate_table.csv`, `direction_decision_table.csv`, `failure_attribution.csv`, family directories and compact `cells/*/*.npz`.',
        'No optional cross-factor combination, additional frequency band, whole-session generalization extension, or tokenizer training was selected from these outer scores.',
        'Next: review complete selection rules and their null/truth checks. Incomplete rules remain diagnostic; retain every failed identity before deciding a new version.']
    lines += ['', 'Completed diagnostic comparisons (each denominator is fixed before inference):', '',
        '| N2 law / processing / full W=0 | Solver | Trials | r NRMSE | EEG / HbO / HbR NRMSE |',
        '|---|---|---:|---:|---|']
    for row in temporal:
        if row['mask'] == 'full' and row['w'] == 0:
            values = row['metrics']
            display = lambda value: 'NOT_ESTIMATED' if value is None else f'{value:.4f}'
            lines.append(f"| {row['law']} / {row['variant']} | {row['solver']} | {row['completed_independent_trials']}/{row['expected_independent_trials']} | "
                         f"{display(values['r']['nrmse'])} | "+' / '.join(display(values[k]['nrmse']) for k in ('clean_EEG','clean_HbO','clean_HbR'))+' |')
    lines += ['', f"N6 mean-driver replay: {family_summary['N6']['r_replay']['completed']}/72 complete, mean gap / training SD = {serial(mean_gap)}.",
        'A replay gap alone is not a physical violation. Matched and mismatch synthetic replay values remain in each N1/N6 synthetic row.']
    if sessions:
        lines += ['', '| N5 subject / session | W curve values | Grid maximum W | Boundary | Half-sample W |', '|---|---:|---:|---|---|']
        for row in sessions:
            lines.append(f"| {row['subject']} / {row['session']} | {row['completed_grid_values']}/{row['expected_grid_values']} | "
                         f"{row.get('estimate_w', 'NOT_ESTIMATED')} | {row.get('boundary', 'NOT_ESTIMATED')} | {row.get('half_sample_w', 'NOT_ESTIMATED')} |")
    path = Path(run_dir)/'OVERNIGHT_REPORT.md'
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    os.replace(temporary, path)
    if final:
        render_examples(run_dir, tasks, results)
    return dict(families=family_summary, candidates=candidates_table, n2_mean_precheck=n2_gate, failures=len(failures))


def prepare_run(run_dir, cfg, dc, base, measured, metadata):
    expected_root = (ROOT/cfg['output_root']).resolve()
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != expected_root or run_dir.exists():
        raise ValueError('prepare requires a fresh direct child of the registered output root')
    checks = engineering_checks(cfg, dc, base)
    inventory = scope_inventory(cfg, base, metadata)
    run_dir.mkdir(parents=True)
    (run_dir/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    atomic_json(run_dir/'scope_inventory.json', inventory)
    atomic_json(run_dir/'fold_inventory.json', {s: [folds(rows, outer) for outer in range(4)] for s, rows in inventory.items()})
    atomic_json(run_dir/'preflight.json', checks)
    manifest = dict(schema=cfg['schema'], execution='preparing', created_at=datetime.now(timezone.utc).isoformat(),
        project_root=ROOT, run_dir=run_dir, teacher_qualification='none',
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        threads={name: os.environ[name] for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    atomic_json(run_dir/'manifest.json', manifest)
    workers = min(3, max(1, len(os.sched_getaffinity(0))-2))
    preparations, pilots = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        jobs = [dict(id=task_id('P0', 'native', s), family='P0', kind='prepare_subject', subject=s, identities=rows)
                for s, rows in inventory.items()]
        pending = {pool.submit(execute_cell, run_dir, cfg, t): t for t in jobs}
        while pending:
            done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                result = future.result()
                append_status(run_dir, result)
                preparations.append(result)
                print(f"Preparation {task['id']}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
        jobs = [dict(id=task_id('P0', 'projection', s, outer), family='P0', kind='prepare_projection', subject=s, outer=outer)
                for s in cfg['subjects'] for outer in (None, 0, 1, 2, 3)]
        pending = {pool.submit(execute_cell, run_dir, cfg, t): t for t in jobs}
        while pending:
            done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                result = future.result()
                append_status(run_dir, result)
                preparations.append(result)
                print(f"Preparation {task['id']}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
        pilot_tasks = [
            dict(id='pilot_N1', family='N1', kind='synthetic', condition='matched_student_t', stream='discovery', replicate=0, candidate=BASE, seed=cfg['seed']+10),
            dict(id='pilot_N2', family='N2', kind='temporal', law='nonlinear_gaussian', replicate=0, variant='model', mode='full', w=0., solver='O2', seed=cfg['seed']+11),
            dict(id='pilot_N3', family='N3', kind='synthetic', condition='matched_student_t', stream='discovery', replicate=0, candidate=dict(BASE, noise=2.), seed=cfg['seed']+10),
            dict(id='pilot_N4', family='N4', kind='artifact', subject=cfg['subjects'][0], eeg='E3', outer=0),
            dict(id='pilot_N5', family='N5', kind='curve', subject=cfg['subjects'][0], candidate=dict(BASE, gain=1.5)),
            dict(id='pilot_N6', family='N6', kind='failure_trace', subject='subject_01', trial=4, w=0.)]
        if 'N7' in cfg:
            pilot_tasks.append(dict(id='pilot_N7', family='N7', kind='synthetic', condition='true_gw',
                stream='discovery', replicate=0, candidate=dict(BASE, id='g-0.3_w-0.25', g=-.3, w=-.25), seed=cfg['seed']+12))
        pending = {pool.submit(execute_cell, run_dir, cfg, t): t for t in pilot_tasks}
        while pending:
            done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                result = future.result()
                append_status(run_dir, result)
                pilots.append(result)
                print(f"Pilot {task['family']}: {result['status']}, {result['elapsed_seconds']:.2f}s, RSS {result['peak_rss_bytes']/1024**2:.0f} MiB", flush=True)
    atomic_json(run_dir/'pilots.json', dict(pilots=pilots, preparations=preparations))
    tasks = make_tasks(cfg, inventory)
    persist_tasks(run_dir, tasks)
    resources = resource_budget(cfg, pilots+preparations)
    manifest.update(execution='prepared', resources=resources, prepared_at=datetime.now(timezone.utc).isoformat(),
        task_count=len(tasks), planned_solves=sum(t['planned_solves'] for t in tasks),
        task_table='task_table.csv', status_ledger='case_status.jsonl')
    atomic_json(run_dir/'manifest.json', manifest)
    summarize_run(run_dir, cfg)
    print(json.dumps(serial(dict(run_dir=run_dir, resources=resources, tasks=len(tasks),
        planned_solves_by_family={f: sum(t['planned_solves'] for t in tasks if t['family'] == f)
                                 for f in (FAMILIES if 'N7' in cfg else FAMILIES[:-1])})), indent=2), flush=True)


def freeze_run(run_dir, cfg):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir/'manifest.json')
    if manifest['execution'] != 'prepared':
        raise ValueError('only an integrated, prepared run can be frozen')
    freeze_sources(run_dir, cfg)
    # Bind the task contract and all fit-fold coordinates before evaluation.
    paths = ['task_table.csv', 'resolved_config.yaml', 'scope_inventory.json', 'fold_inventory.json', 'preflight.json']
    inventory = read_json(run_dir/'scope_inventory.json')
    for subject in inventory:
        for outer in (None, 0, 1, 2, 3):
            for inner in (None,) if outer is None else (None, 0, 1, 2):
                for eeg in ('E0',) if outer is None else cfg['N4']['candidates']:
                    prefix = projection_path(run_dir, subject, outer, inner, eeg)
                    for suffix in ('.json', '.npz'):
                        path = prefix.with_suffix(suffix)
                        if path.exists():
                            paths.append(str(path.relative_to(run_dir)))
    atomic_json(run_dir/'frozen_input_identity.json', {p: diagnostic.digest(run_dir/p) for p in paths})
    manifest.update(execution='ready', frozen_at=datetime.now(timezone.utc).isoformat(),
        snapshot_entrypoint=str(run_dir/'source_snapshot/experiments/evaluate_ssm_overnight_diagnostics.py'))
    atomic_json(run_dir/'manifest.json', manifest)
    print(f"Frozen {len(paths)} task/scope/fold inputs; entrypoint: {manifest['snapshot_entrypoint']}", flush=True)


def run_scheduler(run_dir, cfg):
    run_dir = Path(run_dir).resolve()
    verify_snapshot(run_dir)
    for path, digest in read_json(run_dir/'frozen_input_identity.json').items():
        if diagnostic.digest(run_dir/path) != digest:
            raise ValueError('frozen task/fold contract changed: '+path)
    # flock prevents a second controller from consuming the same queue.
    import fcntl
    lock = (run_dir/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = read_tasks(run_dir)
    manifest = read_json(run_dir/'manifest.json')
    if manifest['execution'] not in ('ready', 'running'):
        raise ValueError('run is not ready/running; completed evidence is immutable')
    pilots = read_json(run_dir/'pilots.json')
    resources = resource_budget(cfg, pilots['pilots']+pilots['preparations'])
    previous_start = manifest.get('started_at')
    started_at = datetime.fromisoformat(previous_start) if previous_start else datetime.now(timezone.utc)
    elapsed = (datetime.now(timezone.utc)-started_at).total_seconds()
    deadline = time.monotonic()+max(0., cfg['budget']['hours']*3600-elapsed)
    manifest.update(execution='running', started_at=started_at.isoformat(), resources=resources,
        controller_pid=os.getpid(), controller_ppid=os.getppid(), controller_session=os.getsid(0),
        service=os.environ.get('SSM_SYSTEMD_UNIT'), execution_source=str(CODE_ROOT))
    atomic_json(run_dir/'manifest.json', manifest)
    terminal = set()
    for task in tasks:
        path = run_dir/'cells'/task['id']/'result.json'
        if path.exists():
            terminal.add(task['id'])
    remaining = deque(t for t in tasks if t['id'] not in terminal)
    running = {}
    stopping = False
    stop_reason = 'all_registered_cells_terminal'

    def stop_handler(signum, frame):
        nonlocal stopping, stop_reason
        stopping, stop_reason = True, f'controller_signal_{signum}'

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    pool = None
    last_update = time.monotonic()
    if resources['workers'] < 1:
        stopping, stop_reason = True, 'insufficient_memory_for_one_job'
    else:
        pool = ProcessPoolExecutor(max_workers=resources['workers'], mp_context=get_context('spawn'))
    print(f"Controller PID={os.getpid()} PPID={os.getppid()} SID={os.getsid(0)} workers={resources['workers']} tasks={len(tasks)} hard budget={cfg['budget']['hours']}h", flush=True)
    try:
        while (remaining or running) and not stopping:
            if time.monotonic() >= deadline:
                stopping, stop_reason = True, 'hard_time_budget'
                break
            available_tokens = resources['memory_tokens_bytes']-sum(v[1] for v in running.values())
            attempts = len(remaining)
            active_layer = min([t['layer'] for t in remaining]+[v[0]['layer'] for v in running.values()], default=3)
            while remaining and len(running) < resources['workers'] and attempts:
                task = remaining.popleft()
                attempts -= 1
                cost = resources['map_job_bytes'] if task.get('solver') == 'O2' else resources['regular_job_bytes']
                if (task['layer'] != active_layer or not set(task['dependencies']).issubset(terminal) or cost > available_tokens):
                    remaining.append(task)
                    continue
                future = pool.submit(execute_cell, run_dir, cfg, task)
                running[future] = (task, cost)
                available_tokens -= cost
            if not running and remaining:
                stopping, stop_reason = True, 'unresolvable_dependency_or_memory_budget'
                break
            done, _ = wait(running, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                task, _ = running.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(task_id=task['id'], family=task['family'], kind=task['kind'], **failure(exc))
                    atomic_json(run_dir/'cells'/task['id']/'result.json', dict(result, rows=partial_rows(run_dir, task)))
                append_status(run_dir, result)
                terminal.add(task['id'])
            if time.monotonic()-last_update >= 60 or (done and len(terminal) <= 12):
                elapsed_hours = (datetime.now(timezone.utc)-started_at).total_seconds()/3600
                print(f"Progress {len(terminal)}/{len(tasks)} cells; running={len(running)}; elapsed={elapsed_hours:.3f}h", flush=True)
                manifest.update(completed_cells=len(terminal), heartbeat_at=datetime.now(timezone.utc).isoformat())
                atomic_json(run_dir/'manifest.json', manifest)
                last_update = time.monotonic()
    except Exception as exc:
        stopping, stop_reason = True, 'controller_exception: '+repr(exc)
        manifest['controller_traceback'] = traceback.format_exc()
    finally:
        if pool is not None:
            if running:
                # Hard deadline applies to active workers too. Their completed
                # per-fit atomic files remain recoverable below.
                processes = list((pool._processes or {}).values())
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=.5)
                    if process.is_alive():
                        process.kill()
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True)
        for task, _ in running.values():
            result = dict(task_id=task['id'], family=task['family'], kind=task['kind'], status='timeout',
                          reason=stop_reason, rows=partial_rows(run_dir, task))
            atomic_json(run_dir/'cells'/task['id']/'result.json', result)
            append_status(run_dir, {k: v for k, v in result.items() if k != 'rows'})
        for task in remaining:
            result = dict(task_id=task['id'], family=task['family'], kind=task['kind'], status='not_started_budget', reason=stop_reason, rows=[])
            atomic_json(run_dir/'cells'/task['id']/'result.json', result)
            append_status(run_dir, {k: v for k, v in result.items() if k != 'rows'})
        # Session bootstrap is cheap aggregation of the frozen grid, including
        # its missing shape, and is useful even when some grid cells timed out.
        for task in tasks:
            if task['kind'] == 'session_summary' and cell_result(run_dir, task['id'])['status'] == 'not_started_budget':
                execute_cell(run_dir, cfg, task)
        manifest.update(execution='completed' if not stopping else 'stopped_budget' if stop_reason == 'hard_time_budget' else 'stopped',
            stop_reason=stop_reason, finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(run_dir/'manifest.json', manifest)
        summary = summarize_run(run_dir, cfg, final=True)
        print(json.dumps(serial(dict(execution=manifest['execution'], stop_reason=stop_reason,
            families=summary['families'], report=str(run_dir/'OVERNIGHT_REPORT.md'))), ensure_ascii=False, indent=2), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-dir', type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    for flag in ('check-only', 'prepare', 'freeze', 'run', 'report'):
        actions.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if args.check_only:
        cfg, dc, base, _, _ = load_config(args.config)
        result = engineering_checks(cfg, dc, base)
        print(json.dumps(serial(result), ensure_ascii=False, indent=2))
        return 0 if all(r['passed'] for r in result.values()) else 1
    if args.run_dir is None:
        parser.error('--run-dir is required')
    if args.prepare:
        cfg, dc, base, measured, metadata = load_config(args.config)
        prepare_run(args.run_dir, cfg, dc, base, measured, metadata)
    else:
        cfg, _, _, _, _ = load_config(args.run_dir/'resolved_config.yaml')
        if args.freeze:
            freeze_run(args.run_dir, cfg)
        elif args.run:
            run_scheduler(args.run_dir, cfg)
        else:
            print(json.dumps(serial(summarize_run(args.run_dir, cfg, final=True)), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
