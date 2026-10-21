#!/usr/bin/env python3
"""Train-frozen local spatial shared-driver development experiment.

The default stage validates software/configuration only. Native public arrays
are read exclusively by ``src.data.shared_driver_spatial`` in the explicit
prepare stage. No stage admits a protected split or arbitrary prepared input.
Every fit, including unavailable/failed fits, has a terminal JSON record; large
numeric outputs stay in its companion NPZ. Scientific verdicts are not inferred
from execution completion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

for _thread_name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '1'

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import yaml

from src.inference.observation_baselines import (
    fit_spatial_coordinates, fit_spatial_eeg_loadings, ridge_fit, ridge_predict,
    spatial_observation_design, transform_spatial_coordinates,
)
from src.inference.shared_driver_spatial import (
    fit_spatial_shared_driver, forward_spatial_shared_driver, temporal_mode_basis,
)
from src.inference.t3a_balloon_robust_ssm import (
    BalloonFixedParameters, BalloonFreeParameters, BalloonParameters,
)

DEFAULT_CONFIG = CODE_ROOT / 'experiments/configs/physiology_semantic_tokenizer/shared_driver_spatial_v1.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_spatial'
STAGES = ('check', 'dryrun', 'synthetic-smoke', 'synthetic', 'plan', 'prepare', 'calibrate', 'measured', 'summarize')
MODES = ('full', 'heldout_pair', 'center_pair')
PAIRINGS = ('paired', 'Hb_only', 'wrong_EEG')
BASELINES = ('training_template', 'spatial_ridge', 'joint_ridge', 'own_context_interpolation')
TERMINAL = ('completed', 'failed_numerical', 'failed_domain', 'failed_physical',
            'failed_integration', 'failed_exception', 'failed_calibration',
            'unavailable_wrong_EEG', 'unavailable_preparation')


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def token(identity):
    return hashlib.sha256(str(identity).encode()).hexdigest()[:24]


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get('schema') != 'shared_driver_spatial_v1'
            or cfg.get('experiment_id') != 'SSM-LOCAL-SPATIAL-v1'
            or cfg.get('artifact_root') != ARTIFACT_ROOT
            or cfg.get('scope') != 'public_development_local_spatial_observation_not_source_localization'):
        raise ValueError('wrong versioned spatial experiment or artifact boundary')
    d, t, m, c, e, s = (cfg[k] for k in ('data', 'tensor', 'model', 'calibration', 'evaluation', 'solver'))
    if (d['datasets'] != ['eeg_fnirs_single_trial'] or d['protected_data'] != 'forbidden'
            or d['cache_root'] != 'data/cache/physiology_semantic_clean_v5'
            or d['region'] != 'prefrontal' or d['hb_pairs'] != 4 or d['eeg_channels'] != 12
            or d['max_patch_angle_deg'] != 45.
            or d['disjoint_optodes'] is not True or d['window_seconds'] != 30.
            or d['window_offset_s'] != -5. or d['reference_seconds'] != 5.
            or d['split'] != 'dataset_subject_disjoint_train_selection_evaluation'
            or d['selection'] != 'metadata_only_nested_patch_and_chronological_nonoverlap'
            or 'geometry_root' in d):
        raise ValueError('public source, geometry, split or window boundary changed')
    if (t['steps'] != 120 or t['dt_s'] != .25 or t['reference_steps'] != 20
            or t['temporal_modes'] != 16 or t['eeg_feature_order'] != 'channel_major_band_minor'
            or t['hb_components'] != ['HbO', 'HbR']
            or t['state_order'] != ['r', 's', 'f', 'v', 'p', 'q']
            or t['eeg_bands_hz'] != [[1, 4], [4, 8], [8, 13], [13, 30], [30, 45]]):
        raise ValueError('frozen tensor, ordering or observation coordinate changed')
    if (m['modes'] != [1, 2] or m['geometry'] != ['true', 'uniform', 'sham']
            or m['main_panel'] != {'hb_pairs': 4, 'eeg_channels': 12}
            or m['count_panels'] != [{'hb_pairs': 2, 'eeg_channels': 12}, {'hb_pairs': 4, 'eeg_channels': 6}]
            or m['gauge'] != 'fixed_physical_driver_beta1_Hb_gain1_no_column_renormalization'
            or m['extra_hb_component'] != 'none' or m['network_edges'] != 'none'
            or c['eeg_weight_candidates'] != [0., .1, 1.]
            or c['eeg_loading_ridge_definition'] != 'ridge_times_mean_diagonal_training_driver_Gram'
            or c['sequence'] != 'train_coordinates_then_Hb_only_drivers_then_signed_EEG_readout_then_selection_subjects'
            or c['selection_mode'] != 'heldout_pair'
            or c['target_coordinate'] != 'same_training_coordinate_for_all_arms_and_nested_panels'):
        raise ValueError('spatial arms, gauge or train/selection sequence changed')
    if (e['modes'] != list(MODES) or e['pairings'] != list(PAIRINGS)
            or e['baselines'] != list(BASELINES) or e['target_pair'] != 0
            or e['center_steps'] != [48, 72] or e['score_steps'] != [20, 120]
            or e['hidden_policy'] != 'mask_before_input_assembly_no_target_reference_PCA_or_window_statistics'
            or e['wrong_EEG'] != 'other_nonoverlapping_record_window_same_subject_task_condition_and_montage'
            or e['bootstrap_unit'] != 'subject_all_windows_targets_and_conditions_together'
            or e['fixed_weight_diagnostic'] != 'main_panels_paired_heldout_and_center_SSM_fixed1'
            or e['tied_diagnostic'] != 'K2_main_paired_heldout_and_center_same_readout_same_weight1_tied_drivers_and_initials'
            or e['failure_penalty_nmse'] <= 0):
        raise ValueError('registered masks, wrong-EEG pairing or endpoint changed')
    fixed = dict(tau=2., kappa=.64, gamma=.32, alpha=.32, E0=.32, P0=1., Q0=.35,
                 neurovascular_gain=1., eeg_loading=1.)
    if cfg['fixed'] != fixed:
        raise ValueError('fixed physiology or physical observation gauge changed')
    if (s['substeps'] != 4 or s['validation_substeps'] != 8
            or s['integration_tolerance_training_sd'] != .005 or s['numerical_backend'] != 'numba'
            or s['max_nfev'] < 1 or s['gradient_tolerance'] <= 0
            or min(s[k] for k in ('driver_curvature_weight', 'initial_weight', 'flow_prior_weight')) < 0
            or s['flow_log_sd'] <= 0 or cfg['resources']['numerical_threads'] != 1
            or not 1 <= cfg['resources']['max_workers'] <= 24
            or cfg['resources']['max_in_flight_per_worker'] != 2):
        raise ValueError('solver or bounded resource contract changed')
    if (not 0 < d['train_fraction'] < 1 or not 0 < d['selection_fraction'] < 1
            or d['train_fraction'] + d['selection_fraction'] >= 1
            or min(d[k] for k in ('max_subjects', 'max_records_per_subject', 'max_windows_per_record')) < 1
            or c['target_rms'] <= 0 or min(c['eeg_loading_ridge'], c['ridge_baseline']) < 0):
        raise ValueError('positive training/split/regularization controls required')
    if (cfg['synthetic']['smoke'] != {'steps': 40, 'temporal_modes': 6, 'repeats': 1}
            or cfg['synthetic']['coordinate_policy'] != 'generator_model_coordinates_identity_scaling_training_only_SD'
            or cfg['synthetic']['records_per_identity'] != 2
            or cfg['synthetic']['common_hb_direction'] != [.65, .35]
            or cfg['synthetic']['scenarios'] != ['single_source', 'two_sources', 'identical_sources',
                'uninformative_EEG', 'geometry_mismatch', 'Hb_common_interference']):
        raise ValueError('synthetic mechanism controls changed')
    return cfg


def parameters(cfg):
    fixed = {k: v for k, v in cfg['fixed'].items() if k not in ('tau', 'kappa')}
    result = BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=cfg['fixed']['tau'], kappa=cfg['fixed']['kappa']))
    result.validate()
    return result


def arm_specs(cfg):
    """Seven registered arms; small panels are K1 true-geometry controls only."""
    arms = [dict(K=k, geometry=g, hb_pairs=4, eeg_channels=12)
            for k, gs in ((1, ('true', 'uniform', 'sham')), (2, ('true', 'sham'))) for g in gs]
    arms += [dict(K=1, geometry='true', **p) for p in cfg['model']['count_panels']]
    for a in arms:
        a['id'] = f'K{a["K"]}_{a["geometry"]}_H{a["hb_pairs"]}_E{a["eeg_channels"]}'
    return arms


def spatial_design(cfg, patch, arm):
    # Freeze the complete-panel kernel width/anchors/norm before prefixing. A
    # small panel cannot covertly refit a new coordinate or renormalize columns.
    design = spatial_observation_design(patch['eeg_coordinates'], patch['hb_coordinates'], arm['K'],
        geometry=arm['geometry'], seed=cfg['model']['sham_seed'])
    design['spatial_eeg'] = design['spatial_eeg'][:arm['eeg_channels']]
    design['hb_loadings'] = design['hb_loadings'][:arm['hb_pairs']]
    return design


def panel_coordinate(coordinate, arm):
    c, j = arm['eeg_channels'], arm['hb_pairs']
    result = dict(coordinate)
    result['eeg_scale'] = np.asarray(coordinate['eeg_scale'])[:c]
    result['eeg_shape'], result['hb_shape'] = (c, 5), (j, 2)
    full_eeg = np.asarray(coordinate['eeg_scale']).size
    result['sd'] = np.r_[np.asarray(coordinate['sd'])[:c*5], np.asarray(coordinate['sd'])[full_eeg:full_eeg+2*j]]
    return result


def visible_mask(cfg, mode, arm, *, steps=None, pairing='paired'):
    if mode not in MODES or pairing not in PAIRINGS:
        raise ValueError('unregistered visibility mode or pairing')
    t, f, j = steps or cfg['tensor']['steps'], arm['eeg_channels']*5, arm['hb_pairs']
    mask = np.ones((t, f + 2*j), bool)
    if mode == 'heldout_pair':
        mask[:, f:f+2] = False
    elif mode == 'center_pair':
        left, right = cfg['evaluation']['center_steps']
        if t != cfg['tensor']['steps']:
            left, right = round(t*left/cfg['tensor']['steps']), round(t*right/cfg['tensor']['steps'])
        mask[left:right, f:f+2] = False
    if pairing == 'Hb_only':
        mask[:, :f] = False
    return mask


def endpoint_mask(cfg, mode, *, steps=None):
    if mode not in MODES:
        raise ValueError('unregistered endpoint')
    t = steps or cfg['tensor']['steps']
    first, last = cfg['evaluation']['score_steps']
    if t != cfg['tensor']['steps']:
        first, last = round(t*first/cfg['tensor']['steps']), t
    result = np.zeros((t, 2), bool)
    result[first:last] = True
    if mode == 'center_pair':
        left, right = cfg['evaluation']['center_steps']
        if t != cfg['tensor']['steps']:
            left, right = round(t*left/cfg['tensor']['steps']), round(t*right/cfg['tensor']['steps'])
        result[:left] = False
        result[right:] = False
    return result


def assemble_inputs(eeg, hb, mask):
    """Mask before materializing fitter/baseline inputs; never inspect hidden values."""
    eeg, hb = np.asarray(eeg, float), np.asarray(hb, float)
    f = eeg.shape[1]
    if mask.shape != (len(eeg), f + hb.shape[1]*2) or mask.dtype != np.bool_:
        raise ValueError('input mask shape/dtype mismatch')
    e = np.where(mask[:, :f], eeg, np.nan)
    h = np.where(mask[:, f:].reshape(hb.shape), hb, np.nan)
    if not np.isfinite(e[mask[:, :f]]).all() or not np.isfinite(h[mask[:, f:].reshape(hb.shape)]).all():
        raise ValueError('nonfinite visible input')
    return e, h


def wrong_eeg_partners(rows):
    """Deterministic within-subject/condition/montage nonoverlapping pairing."""
    partners = {}
    for row in rows:
        candidates = []
        for other in rows:
            if row['id'] == other['id'] or any(row.get(k) != other.get(k) for k in ('dataset', 'subject', 'task', 'condition', 'split')):
                continue
            if any(row['patch'].get(k) != other['patch'].get(k) for k in
                   ('eeg_channel_names', 'hb_pair_names', 'coordinate_system', 'eeg_coordinates', 'hb_coordinates')):
                continue
            if row['record'] == other['record']:
                if any(abs(row[k] - other[k]) < row['window_seconds'] for k in ('eeg_start_s', 'hb_start_s')):
                    continue
            candidates.append(other['id'])
        partners[row['id']] = min(candidates) if candidates else None
    return partners


def validate_subject_groups(rows):
    identity, groups = set(), {}
    for row in rows:
        if row['id'] in identity or row['split'] not in ('train', 'selection', 'evaluation'):
            raise ValueError('duplicate identity or unregistered subject partition')
        identity.add(row['id'])
        key = (row['dataset'], row['subject'])
        if key in groups and groups[key] != row['split']:
            raise ValueError('subject leaked between train, selection and evaluation')
        groups[key] = row['split']
    if set(groups.values()) != {'train', 'selection', 'evaluation'}:
        raise ValueError('nonempty train, selection and evaluation subject groups required')
    return groups


def check_geometry(rows):
    keys = ('eeg_channel_names', 'hb_pair_names', 'eeg_coordinates', 'hb_coordinates', 'coordinate_system')
    reference = rows[0]['patch']
    if any(any(row['patch'].get(k) != reference.get(k) for k in keys) for row in rows):
        raise ValueError('v1 frozen signed-band calibration requires one shared verified montage')
    return reference


def plan_manifest(cfg, inventory):
    rows = inventory['rows']
    unavailable = inventory.get('status') == 'unavailable'
    try:
        if not unavailable:
            validate_subject_groups(rows)
            check_geometry(rows)
    except ValueError as exc:
        unavailable = True
        inventory = dict(inventory, unavailable_reason=str(exc))
    if unavailable:
        return dict(schema='shared_driver_spatial_plan_v1', status='unavailable', rows=rows,
            excluded=inventory.get('excluded', []), inventory=inventory,
            subjects=inventory.get('subjects'), cells=[], planned_cells=0,
            scientific_verdict='not_assigned', measured_arrays_read=0,
            reason=inventory.get('unavailable_reason', 'insufficient_verified_public_spatial_support'))
    partners = wrong_eeg_partners(rows)
    cells = []
    for row in rows:
        if row['split'] != 'evaluation':
            continue
        for arm in arm_specs(cfg):
            for mode in MODES:
                for pairing in PAIRINGS:
                    methods = ('SSM', *BASELINES)
                    if arm['hb_pairs'] == 4 and arm['eeg_channels'] == 12 and mode != 'full' and pairing == 'paired':
                        methods += ('SSM_fixed1',)
                        if arm['K'] == 2:
                            methods += ('SSM_tied_fixed1',)
                    for method in methods:
                        identity = '|'.join((row['id'], arm['id'], mode, pairing, method))
                        cells.append(dict(id=token(identity), record_id=row['id'], dataset=row['dataset'],
                            subject=row['subject'], arm=arm['id'], mode=mode, pairing=pairing, method=method,
                            scenario=row.get('scenario', 'public_development'),
                            wrong_eeg_id=partners[row['id']] if pairing == 'wrong_EEG' else None))
    return dict(schema='shared_driver_spatial_plan_v1', status='available', rows=rows, excluded=inventory.get('excluded', []),
                inventory=inventory,
                subjects=inventory.get('subjects'), wrong_eeg_partners=partners, cells=cells,
                planned_cells=len(cells), scientific_verdict='not_assigned', measured_arrays_read=0)


def _prepared_path(out, identity):
    return Path(out) / 'prepared' / (token(identity) + '.npz')


def load_prepared(out, row):
    """Only the immutable run's own frozen-plan identity can select an array."""
    path = _prepared_path(out, row['id'])
    meta = read_json(path.with_suffix('.json'))
    if meta.get('status') != 'completed' or meta.get('row') != row:
        raise ValueError('missing, failed or changed frozen prepared identity')
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


def _prepare_job(payload):
    cfg, root, out, row = payload
    path = _prepared_path(out, row['id'])
    if path.with_suffix('.json').exists():
        return read_json(path.with_suffix('.json'))
    try:
        from src.data.shared_driver_spatial import prepare_spatial_record
        record = prepare_spatial_record(cfg, root, row)
        write_npz(path, eeg=record['eeg'], hb=record['hb'], input_hb=record['input_hb'],
                  eeg_operator=record['operators']['eeg'], hb_operator=record['operators']['model'])
        result = dict(status='completed', row=row, provenance=record['metadata'])
    except Exception as exc:
        result = dict(status='failed_exception', row=row, error=str(exc), traceback=traceback.format_exc())
    write_json(path.with_suffix('.json'), result)
    return result


def _panel_arrays(record, coordinate, arm):
    return transform_spatial_coordinates(record['eeg'][:, :arm['eeg_channels']],
        record['hb'][:, :arm['hb_pairs']], panel_coordinate(coordinate, arm))


def masked_panel_arrays(record, coordinate, arm, mask, *, wrong_eeg=None):
    """Apply the information boundary in native feature coordinates first."""
    c, j = arm['eeg_channels'], arm['hb_pairs']
    # A wrong-EEG donor supplies EEG only; donor Hb is never transformed,
    # validated, summarized, or passed to the target window's model.
    source = record if wrong_eeg is None else wrong_eeg
    raw_e = np.asarray(source['eeg'])[:, :c]
    raw_h = np.asarray(record['hb'])[:, :j]
    e = np.where(mask[:, :c*5].reshape(raw_e.shape), raw_e, np.nan)
    h = np.where(mask[:, c*5:].reshape(raw_h.shape), raw_h, np.nan)
    e, h = transform_spatial_coordinates(e, h, panel_coordinate(coordinate, arm))
    return assemble_inputs(e, h, mask)


def scoring_target(record, coordinate):
    """A separate scoring-only path; never passed into input construction."""
    return np.asarray(record['hb'], float)[:, 0]/float(coordinate['hb_scale'])


def _forward_kwargs(cfg, record, design, loading, coordinate):
    steps = len(record['eeg'])
    modes = cfg['tensor']['temporal_modes'] if steps == cfg['tensor']['steps'] else cfg['synthetic']['smoke']['temporal_modes']
    return dict(eeg_loadings=np.asarray(loading), hb_loadings=np.asarray(design['hb_loadings']),
        temporal_basis=temporal_mode_basis(steps, modes),
        hb_gains=np.ones(len(design['hb_loadings'])),
        eeg_operator=record['eeg_operator'], hb_operator=record['hb_operator'],
        substeps=cfg['solver']['substeps'], numerical_backend=cfg['solver']['numerical_backend'])


def fit_record(cfg, record, coordinate, arm, design, loading, mode, pairing, weight, *, wrong=None, tied=False):
    if pairing == 'wrong_EEG' and wrong is None:
        raise ValueError('wrong EEG requires a matched nonoverlapping prepared record')
    mask = visible_mask(cfg, mode, arm, steps=len(record['eeg']), pairing=pairing)
    if weight == 0:
        mask[:, :arm['eeg_channels']*5] = False
    e, h = masked_panel_arrays(record, coordinate, arm, mask,
                               wrong_eeg=wrong if pairing == 'wrong_EEG' else None)
    s = cfg['solver']
    forward = _forward_kwargs(cfg, record, design, loading, coordinate)
    fitter = fit_spatial_shared_driver
    if tied:
        from src.inference.shared_driver_spatial import fit_tied_spatial_shared_driver
        fitter = fit_tied_spatial_shared_driver
    fit = fitter(e, h, parameters(cfg), cfg['tensor']['dt_s'],
        **forward, sd=panel_coordinate(coordinate, arm)['sd'], visible=mask,
        eeg_weight=0. if pairing == 'Hb_only' else weight,
        driver_curvature_weight=s['driver_curvature_weight'], initial_weight=s['initial_weight'],
        flow_prior_weight=s['flow_prior_weight'], flow_log_sd=s['flow_log_sd'],
        max_nfev=s['max_nfev'], gradient_tolerance=s['gradient_tolerance'])
    # Validate the accepted physical trajectory against a finer integration.
    # This check contributes to primary success, including calibration fits.
    if fit.get('success'):
        replay_args = dict(forward, substeps=s['validation_substeps'], derivative=False)
        if tied:
            from src.inference.shared_driver_spatial import collapse_tied_spatial_design
            replay_args.update(collapse_tied_spatial_design(forward['eeg_loadings'], forward['hb_loadings']))
        try:
            replay = forward_spatial_shared_driver(fit['coefficients'], fit['initial_coordinates'],
                parameters(cfg), cfg['tensor']['dt_s'], **replay_args)
            sd = np.asarray(panel_coordinate(coordinate, arm)['sd'])
            error = float(np.max(np.abs(replay['prediction'] - fit['prediction'])/sd))
        except Exception as exc:
            error = float('inf')
            fit['integration_error_message'] = str(exc)
        fit['integration_max_training_sd'] = error
        if not error <= s['integration_tolerance_training_sd']:
            fit['success'], fit['status'] = False, 'failed_integration'
    fit['visible'] = mask
    return fit


def score_prediction(prediction, target, sd, endpoint):
    p, y, sd = np.asarray(prediction, float), np.asarray(target, float), np.asarray(sd, float)
    if p.shape != y.shape or y.shape != endpoint.shape or y.shape[1] != 2 or sd.shape != (2,):
        raise ValueError('pair score shape mismatch')
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError('positive finite frozen scoring SD required')
    if not endpoint.any() or not np.isfinite(p[endpoint]).all() or not np.isfinite(y[endpoint]).all():
        raise ValueError('finite nonempty target endpoint required')
    per_component, correlations, ratios = [], [], []
    for index in range(2):
        take = endpoint[:, index]
        yp, yt = p[take, index], y[take, index]
        per_component.append(float(np.mean(((yp-yt)/sd[index])**2)))
        correlations.append(float(np.corrcoef(yp, yt)[0, 1]) if yp.std() > 0 and yt.std() > 0 else None)
        denominator = float(np.sqrt(np.mean(yt**2)))
        ratios.append(float(np.sqrt(np.mean(yp**2))/denominator) if denominator > 0 else None)
    nmse = float(np.mean(per_component))
    if not np.isfinite(nmse):
        raise ValueError('nonfinite primary endpoint')
    return dict(nmse=nmse, nrmse=np.sqrt(nmse), component_nmse=per_component,
                correlation=correlations, amplitude_rms_ratio=ratios,
                scored_values=int(endpoint.sum()), denominator='frozen_training_SD_per_component',
                pooling='mean_time_then_equal_HbO_HbR; subject_aggregation_precedes_square_root')


def baseline_inputs(eeg, hb, method, pairing):
    """Contemporaneous donor pairs, optionally EEG; target is never indexed."""
    if method not in BASELINES or pairing not in PAIRINGS:
        raise ValueError('unregistered baseline or pairing')
    e, h = np.asarray(eeg, float), np.asarray(hb, float)
    donors = h[:, 1:, :].reshape(len(h), -1)
    if method in ('training_template', 'own_context_interpolation'):
        return np.empty((len(e), 0))
    x = np.column_stack((e, donors)) if method == 'joint_ridge' and pairing != 'Hb_only' else donors
    if not np.isfinite(x).all():
        raise ValueError('baseline uses finite visible EEG/donors only')
    return x


def fit_baselines(chunks, cfg, training_ids):
    template = np.mean([h[:, 0] for _, h in chunks], axis=0)
    target = np.concatenate([h[:, 0] for _, h in chunks])
    models = dict(training_template=dict(template=template, training_ids=training_ids))
    for method in ('spatial_ridge', 'joint_ridge'):
        for pairing in ('paired', 'Hb_only'):
            x = np.concatenate([baseline_inputs(e, h, method, pairing) for e, h in chunks])
            models[method+'__'+pairing] = dict(ridge_fit(x, target, cfg['calibration']['ridge_baseline']),
                training_ids=training_ids, inputs='donor_Hb_pairs'+('_and_EEG' if method == 'joint_ridge' and pairing == 'paired' else ''),
                target_pair_excluded_at_all_times=True)
    return models


def predict_baseline(models, eeg, hb, method, pairing):
    if method == 'training_template':
        return np.asarray(models[method]['template'], float).copy()
    key = method+'__'+('Hb_only' if pairing == 'Hb_only' else 'paired')
    model = {k: np.asarray(v) if isinstance(v, list) else v for k, v in models[key].items()}
    return ridge_predict(model, baseline_inputs(eeg, hb, method, pairing))


def predict_own_context(template, hb, target_visible):
    """Interpolate only visible target context; absent context uses train template."""
    target = np.asarray(hb, float)[:, 0]
    result = np.asarray(template, float).copy()
    if target_visible.shape != target.shape or result.shape != target.shape:
        raise ValueError('own-context mask/template shape mismatch')
    time_grid = np.arange(len(target))
    for index in range(2):
        take = target_visible[:, index]
        if take.any():
            if not np.isfinite(target[take, index]).all():
                raise ValueError('nonfinite visible target context')
            result[:, index] = np.interp(time_grid, time_grid[take], target[take, index])
    return result


def choose_eeg_weight(selection_rows, candidates, penalty):
    """Equal subjects; failed/unavailable cells retain the registered penalty."""
    scored = []
    for weight in candidates:
        per_subject = {}
        rows = [r for r in selection_rows if r['eeg_weight'] == weight]
        for row in rows:
            value = row['nmse'] if valid_score(row) else penalty
            per_subject.setdefault(row['subject'], []).append(float(value))
        if not per_subject:
            raise ValueError('every selection candidate needs planned subjects')
        means = {subject: float(np.mean(values)) for subject, values in per_subject.items()}
        scored.append(dict(eeg_weight=weight, subject_mean_nmse=float(np.mean(list(means.values()))),
            subject_nmse=means, planned=len(rows), succeeded=sum(valid_score(r) for r in rows)))
    chosen = min(scored, key=lambda r: (r['subject_mean_nmse'], r['eeg_weight']))
    return dict(selected_eeg_weight=chosen['eeg_weight'], candidates=scored,
                selection_unit='subject', failure_penalty_nmse=penalty)


def _calibrate_job(payload):
    cfg, out, rows, arm = payload
    path = Path(out)/'calibration'/(arm['id']+'.json')
    if path.exists():
        return read_json(path)
    result = dict(arm=arm, status='failed_calibration', training_fits=[], selection_fits=[])
    try:
        coordinate = read_json(Path(out)/'coordinate.json')
        design = spatial_design(cfg, check_geometry(rows), arm)
        training = [r for r in rows if r['split'] == 'train']
        selection = [r for r in rows if r['split'] == 'selection']
        chunks, driver_chunks, eeg_chunks, used_ids = [], [], [], []
        placeholder = np.repeat(np.asarray(design['spatial_eeg']), 5, axis=0)
        for row in training:
            record = load_prepared(out, row)
            e, h = _panel_arrays(record, coordinate, arm)
            chunks.append((e, h))
            fit = fit_record(cfg, record, coordinate, arm, design, placeholder, 'full', 'Hb_only', 0.)
            result['training_fits'].append(dict(record_id=row['id'], subject=row['subject'], **fit_metadata(fit)))
            if fit['success']:
                eeg_chunks.append(e)
                # EEG readout acts on processed drivers, matching the actual
                # training log-power reference operator, rather than raw r(t).
                driver_chunks.append(record['eeg_operator'] @ fit['drivers'])
                used_ids.append(row['id'])
        result['baselines'] = fit_baselines(chunks, cfg, [r['id'] for r in training])
        result['design'] = design
        result['training_subset'] = dict(planned=len(training), succeeded=len(used_ids), ids=used_ids,
            interpretation='successful_conditional_Hb_estimates_only_not_neural_truth')
        try:
            if not used_ids:
                raise ValueError('no successful Hb-only training trajectory for signed EEG calibration')
            readout = fit_spatial_eeg_loadings(eeg_chunks, driver_chunks, design['spatial_eeg'],
                ridge=cfg['calibration']['eeg_loading_ridge'], training_ids=used_ids)
            result['readout'] = readout
            result['eeg_readout_status'] = 'learned'
            loading = readout['eeg_loadings']
        except (ValueError, np.linalg.LinAlgError) as exc:
            # A missing EEG readout cannot invalidate the independent Hb-only
            # path. The geometric matrix is shape metadata only at weight 0;
            # it never enters a data residual or warm start.
            result['readout'] = dict(status='unavailable', error=str(exc), training_ids=used_ids)
            result['eeg_readout_status'] = 'unavailable'
            result['zero_weight_placeholder_loading'] = placeholder
            loading = placeholder
        for weight in cfg['calibration']['eeg_weight_candidates']:
            for row in selection:
                entry = dict(record_id=row['id'], subject=row['subject'], eeg_weight=weight, success=False)
                if weight > 0 and result['eeg_readout_status'] != 'learned':
                    entry.update(status='failed_calibration', reason='EEG_readout_unavailable_positive_weight_not_defined')
                    result['selection_fits'].append(entry)
                    continue
                try:
                    record = load_prepared(out, row)
                    fit = fit_record(cfg, record, coordinate, arm, design, loading,
                        'heldout_pair', 'paired', weight)
                    entry.update(fit_metadata(fit))
                    if fit['success']:
                        f = arm['eeg_channels']*5
                        target = scoring_target(record, coordinate)
                        entry.update(score_prediction(fit['hb_prediction'][:, 0], target,
                            np.asarray(panel_coordinate(coordinate, arm)['sd'])[f:f+2],
                            endpoint_mask(cfg, 'heldout_pair', steps=len(target))))
                except Exception as exc:
                    entry.update(status='failed_exception', success=False, error=str(exc))
                result['selection_fits'].append(entry)
        result.update(choose_eeg_weight(result['selection_fits'], cfg['calibration']['eeg_weight_candidates'],
            cfg['evaluation']['failure_penalty_nmse']))
        result.update(status='completed', calibration='learned_train_only_then_frozen_selection' if
            result['eeg_readout_status'] == 'learned' else 'train_only_coordinates_Hb_geometry_zero_weight_no_learned_EEG_readout',
            physical_hb_gain=1., hb_coordinate_gain=1.,
            gauge='fixed_Hb_gain1_conditional_scaled_target_not_absolute_calibration',
            evaluation_data_used=False, baselines_independent_of_SSM_success=True)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        # Baselines are fitted before the SSM readout and retained even when
        # calibration fails; the evaluator does not inherit the SSM status.
    write_json(path, result)
    return result


def fit_metadata(fit):
    keys = ('success', 'status', 'converged', 'convergence_reason', 'nfev', 'iterations', 'objective',
            'objective_components', 'physical_checks', 'support', 'complexity', 'failure_log',
            'integration_max_training_sd', 'integration_error_message', 'scaled_gradient_inf_norm',
            'original_K', 'effective_K', 'tied', 'objective_exactness', 'regularization_multiplicity')
    return {k: fit[k] for k in keys if k in fit}


def _evaluate_job(payload):
    cfg, out, row, spec = payload
    path = Path(out)/'cells'/(spec['id']+'.json')
    if path.exists():
        return read_json(path)
    result = dict(spec, success=False)
    try:
        arm = next(a for a in arm_specs(cfg) if a['id'] == spec['arm'])
        coordinate = read_json(Path(out)/'coordinate.json')
        calibration = read_json(Path(out)/'calibration'/(arm['id']+'.json'))
        if spec['pairing'] == 'wrong_EEG' and spec['wrong_eeg_id'] is None:
            result['status'] = 'unavailable_wrong_EEG'
        else:
            plan = read_json(Path(out)/'plan.json')
            lookup = {r['id']: r for r in plan['rows']}
            record = load_prepared(out, row)
            wrong = load_prepared(out, lookup[spec['wrong_eeg_id']]) if spec['pairing'] == 'wrong_EEG' else None
            if spec['method'] in ('SSM', 'SSM_fixed1', 'SSM_tied_fixed1'):
                weight = 0. if spec['pairing'] == 'Hb_only' else calibration.get('selected_eeg_weight') if spec['method'] == 'SSM' else 1.
                if calibration['status'] != 'completed':
                    result['status'] = 'failed_calibration'
                elif weight > 0 and calibration.get('eeg_readout_status', 'learned') != 'learned':
                    result.update(status='failed_calibration', reason='EEG_readout_unavailable_positive_weight_not_defined',
                                  eeg_readout_status='unavailable')
                else:
                    loading = calibration['readout']['eeg_loadings'] if calibration.get('eeg_readout_status', 'learned') == 'learned' else calibration['zero_weight_placeholder_loading']
                    fit = fit_record(cfg, record, coordinate, arm, calibration['design'],
                        loading, spec['mode'], spec['pairing'], weight, wrong=wrong,
                        tied=spec['method'] == 'SSM_tied_fixed1')
                    result.update(fit_metadata(fit))
                    result['eeg_weight'] = weight
                    result['eeg_ignored'] = result['eeg_weight'] == 0.
                    result['eeg_readout_status'] = calibration.get('eeg_readout_status', 'learned')
                    result['tied_diagnostic'] = spec['method'] == 'SSM_tied_fixed1'
                    if result['tied_diagnostic']:
                        result['restriction'] = 'same_frozen_K2_readout_tied_drivers_and_initials_vs_two_independent_trajectories_and_initials'
                    prediction = fit['hb_prediction'][:, 0]
                    write_npz(path.with_suffix('.npz'), prediction=fit['prediction'],
                        drivers=fit['drivers'], states=fit['states'], coefficients=fit['coefficients'],
                        initial_coordinates=fit['initial_coordinates'], visible=fit['visible'])
                    if 'truth_drivers' in record:
                        result['known_truth_diagnostics'] = truth_diagnostics(fit, record)
            else:
                mask = visible_mask(cfg, spec['mode'], arm, steps=len(record['eeg']), pairing=spec['pairing'])
                if spec['method'] != 'own_context_interpolation':
                    # Ridge/template controls exclude the entire target pair,
                    # even for full/center reconstruction diagnostics.
                    mask[:, arm['eeg_channels']*5:arm['eeg_channels']*5+2] = False
                if spec['method'] == 'training_template':
                    mask[:] = False
                elif spec['method'] == 'own_context_interpolation':
                    mask[:, :arm['eeg_channels']*5] = False
                    mask[:, arm['eeg_channels']*5+2:] = False
                elif spec['method'] == 'spatial_ridge':
                    mask[:, :arm['eeg_channels']*5] = False
                e, h = masked_panel_arrays(record, coordinate, arm, mask, wrong_eeg=wrong)
                if spec['method'] == 'own_context_interpolation':
                    prediction = predict_own_context(calibration['baselines']['training_template']['template'], h,
                        mask[:, arm['eeg_channels']*5:arm['eeg_channels']*5+2])
                else:
                    prediction = predict_baseline(calibration['baselines'], e, h, spec['method'], spec['pairing'])
                result.update(success=True, status='completed', independent_of_SSM_success=True,
                    baseline_input='training_template' if spec['method'] == 'training_template' else
                        'visible_own_target_context_or_training_template' if spec['method'] == 'own_context_interpolation' else
                        'other_Hb_pairs_only' if spec['method'] == 'spatial_ridge' or spec['pairing'] == 'Hb_only' else
                        'visible_EEG_and_other_Hb_pairs',
                    target_pair_excluded_at_all_times=spec['method'] != 'own_context_interpolation',
                    same_input_comparison=spec['mode'] == 'heldout_pair' or spec['method'] == 'own_context_interpolation')
                write_npz(path.with_suffix('.npz'), target_prediction=prediction)
            if result.get('success'):
                f = arm['eeg_channels']*5
                target = scoring_target(record, coordinate)
                result.update(score_prediction(prediction, target,
                    np.asarray(panel_coordinate(coordinate, arm)['sd'])[f:f+2],
                    endpoint_mask(cfg, spec['mode'], steps=len(target))))
    except Exception as exc:
        result.update(status='failed_exception', success=False, error=str(exc), traceback=traceback.format_exc())
    write_json(path, result)
    return result


def parallel_stage(worker, payloads, workers, out, stage, *, in_flight=2):
    """Bound queued work and persist progress after every terminal cell."""
    started, completed, failures = time.monotonic(), 0, 0
    payloads = iter(payloads)
    def record(value):
        nonlocal completed, failures
        completed += 1
        failures += value.get('status') != 'completed'
        write_json(Path(out)/(stage+'_progress.json'), dict(stage=stage, terminal=completed,
            noncompleted=failures, elapsed_s=time.monotonic()-started, workers=workers,
            max_in_flight=workers*in_flight, numerical_threads=1))
    if workers == 1:
        for payload in payloads:
            record(worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = set()
            while True:
                while len(pending) < workers*in_flight:
                    try:
                        pending.add(pool.submit(worker, next(payloads)))
                    except StopIteration:
                        break
                if not pending:
                    break
                ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    record(future.result())
    return dict(stage=stage, terminal=completed, noncompleted=failures, elapsed_s=time.monotonic()-started)


def fit_coordinate(out, rows, cfg):
    path = Path(out)/'coordinate.json'
    training = [r for r in rows if r['split'] == 'train']
    ids = [r['id'] for r in training]
    if path.exists():
        coordinate = read_json(path)
        if coordinate['training_ids'] != ids:
            raise ValueError('frozen coordinate training identities changed')
        return coordinate
    chunks = []
    for row in training:
        record = load_prepared(out, row)
        chunks.append((record['eeg'], record['hb']))
    if all(row['dataset'] == 'synthetic' for row in training):
        # Known-truth mechanisms are generated directly in the declared model
        # coordinate. Identity observation scales retain their driver/state
        # gauge; only scoring SD is estimated from the training partition.
        eeg = np.concatenate([e.reshape(len(e), -1) for e, _ in chunks])
        hb = np.concatenate([h.reshape(len(h), -1) for _, h in chunks])
        if eeg.shape[1] != 60 or hb.shape[1] != 8 or not np.isfinite(eeg).all() or not np.isfinite(hb).all():
            raise ValueError('finite complete synthetic training panel required')
        sd_before_floor = np.column_stack((eeg, hb)).std(axis=0)
        floor = cfg['calibration']['target_rms']*1e-6
        coordinate = dict(eeg_scale=np.ones((12, 5)), hb_scale=1., eeg_shape=(12, 5), hb_shape=(4, 2),
            training_ids=ids, training_count=len(chunks), training_rows=len(eeg),
            target_rms=cfg['calibration']['target_rms'], sd_floor=floor,
            sd=np.maximum(sd_before_floor, floor), sd_before_floor=sd_before_floor,
            sd_floor_applied=sd_before_floor < floor,
            coordinate_policy=cfg['synthetic']['coordinate_policy'],
            eeg_scale_definition='generator_model_coordinate_identity_no_data_fit',
            centering='none_already_referenced_inputs')
    else:
        coordinate = fit_spatial_coordinates(chunks, ids, target_rms=cfg['calibration']['target_rms'])
    coordinate['training_subjects'] = sorted({r['subject'] for r in training})
    coordinate['fit_scope'] = 'complete_4Hb_12EEG_training_panel_once_shared_all_arms'
    write_json(path, coordinate)
    return coordinate


def _synthetic_patch():
    angles = np.linspace(-1.1, 1.1, 12)
    eeg = np.column_stack((np.sin(angles), np.cos(angles), .2*np.cos(2*angles)))
    hb = np.asarray([[-.8, .6, .1], [-.25, .97, .15], [.35, .94, .2], [.85, .53, .1]])
    return dict(eeg_coordinates=eeg.tolist(), hb_coordinates=hb.tolist(),
        eeg_channel_names=[f'E{i:02}' for i in range(12)], hb_pair_names=[f'H{i}' for i in range(4)],
        coordinate_system='synthetic_normalized_head', coordinate_units='normalized_head_unit')


def synthetic_record(cfg, scenario, split, repeat, *, smoke=False, record_index=0):
    settings = cfg['synthetic']
    index = settings['scenarios'].index(scenario)
    rng = np.random.default_rng(settings[split+'_seed'] + 1009*index + repeat + 1000003*record_index)
    t = settings['smoke']['steps'] if smoke else cfg['tensor']['steps']
    m = settings['smoke']['temporal_modes'] if smoke else cfg['tensor']['temporal_modes']
    k = 1 if scenario == 'single_source' else 2
    patch = _synthetic_patch()
    design = spatial_observation_design(patch['eeg_coordinates'], patch['hb_coordinates'], k,
        geometry='sham' if scenario == 'geometry_mismatch' else 'true', seed=cfg['model']['sham_seed'])
    basis = temporal_mode_basis(t, m)
    coefficient = rng.normal(size=(m, k))/np.arange(1, m+1)[:, None]**1.5
    drivers = basis@coefficient
    coefficient *= settings['driver_rms']/np.sqrt(np.mean(drivers**2, axis=0))
    if scenario == 'identical_sources':
        coefficient[:, 1] = coefficient[:, 0]
    bands = np.asarray([[1., -.7], [-.6, 1.], [1.2, .8], [.4, -.8], [-.8, .5]])[:, :k]
    loading = (np.asarray(design['spatial_eeg'])[:, None, :]*bands[None]).reshape(60, k)
    reference = min(cfg['tensor']['reference_steps'], max(2, round(t/6)))
    operator = np.eye(t)-np.ones((t, 1))@np.r_[np.full(reference, 1/reference), np.zeros(t-reference)][None]
    result = forward_spatial_shared_driver(coefficient, np.zeros((k, 5)), parameters(cfg), cfg['tensor']['dt_s'],
        eeg_loadings=loading, hb_loadings=design['hb_loadings'], temporal_basis=basis,
        eeg_operator=operator, hb_operator=operator, derivative=False,
        substeps=cfg['solver']['validation_substeps'], numerical_backend=cfg['solver']['numerical_backend'])
    e, h = result['eeg_prediction'].copy(), result['hb_prediction'].copy()
    if scenario == 'uninformative_EEG':
        unrelated_coefficients = rng.normal(size=(m, k))/np.arange(1, m+1)[:, None]**1.5
        unrelated = basis@unrelated_coefficients
        unrelated *= settings['driver_rms']/np.sqrt(np.mean(unrelated**2, axis=0))
        e = operator@unrelated@loading.T
    noise = settings['noise_sd']
    e += operator@rng.normal(scale=noise['eeg'], size=e.shape)
    h += np.einsum('st,tjc->sjc', operator,
        rng.normal(size=h.shape)*np.asarray([noise['HbO'], noise['HbR']]))
    if scenario == 'Hb_common_interference':
        interference = .002*np.sin(np.linspace(0, 2*np.pi, t)+.3)
        h += (operator@interference)[:, None, None]*np.asarray(settings['common_hb_direction'])
    identity = f'synthetic|{scenario}|{split}|{repeat}|record={record_index}'
    # Two records share a subject/task/condition/montage but use independent
    # coefficients/noise, providing a genuine matched wrong-EEG control.
    row = dict(id=identity, dataset='synthetic', subject=f'{split}_{repeat:02}',
        record=identity, record_index=record_index, split=split, scenario=scenario, task=scenario, condition='fixed_generator',
        eeg_start_s=0., hb_start_s=0., window_seconds=t*cfg['tensor']['dt_s'], patch=patch)
    return row, dict(eeg=e.reshape(t, 12, 5), hb=h,
        input_hb=np.where(np.arange(4)[None, :, None] == 0, np.nan, h),
        eeg_operator=operator, hb_operator=operator, truth_drivers=result['drivers'],
        truth_states=result['states'], truth_coefficients=coefficient,
        truth_eeg_loadings=loading, truth_hb_loadings=design['hb_loadings'])


def truth_diagnostics(fit, record):
    result = dict(coordinate='generator_model_coordinate_identity',
        comparison='fixed_anchor_order_no_posthoc_sign_permutation_or_scale_fit',
        reference_steps=min(20, max(2, round(len(record['eeg'])/6))),
        state_order=['r', 's', 'f', 'v', 'p', 'q'])
    reference = result['reference_steps']
    def coordinate_scores(actual, truth):
        if not np.isfinite(actual).all() or not np.isfinite(truth).all():
            return dict(status='nonfinite_fit_or_truth')
        actual_mean, truth_mean = float(actual[:reference].mean()), float(truth[:reference].mean())
        centered_actual, centered_truth = actual-actual_mean, truth-truth_mean
        raw_rmse = float(np.sqrt(np.mean((actual-truth)**2)))
        centered_rmse = float(np.sqrt(np.mean((centered_actual-centered_truth)**2)))
        truth_centered_rms = float(np.sqrt(np.mean(centered_truth**2)))
        return dict(status='computed', raw_rmse=raw_rmse, reference_centered_rmse=centered_rmse,
            reference_centered_nrmse=centered_rmse/truth_centered_rms if truth_centered_rms > 1e-12 else None,
            truth_reference_centered_rms=truth_centered_rms,
            reference_dc_error=actual_mean-truth_mean,
            undefined_normalization='None_when_truth_reference_centered_RMS_is_zero')
    for name in ('drivers', 'states'):
        actual, truth = np.asarray(fit[name]), np.asarray(record['truth_'+name])
        if fit.get('tied') and actual.shape[1] == 1 and truth.shape[1] == 2:
            actual = np.repeat(actual, 2, axis=1)
        if actual.shape != truth.shape:
            result[name] = dict(status='unavailable_different_number_of_cores',
                                fitted_shape=actual.shape, truth_shape=truth.shape)
            continue
        if name == 'drivers':
            result[name] = [dict(mode=index, **coordinate_scores(actual[:, index], truth[:, index]))
                            for index in range(actual.shape[1])]
        else:
            result[name] = [dict(mode=index, coordinates={state: coordinate_scores(actual[:, index, column], truth[:, index, column])
                for column, state in enumerate(result['state_order'])}) for index in range(actual.shape[1])]
    return result


def synthetic_stage(cfg, out, workers=1):
    """Separate learned train/selection/evaluation calibration per mechanism."""
    children = []
    for scenario in cfg['synthetic']['scenarios']:
        child = Path(out)/'scenarios'/scenario
        if (child/'plan.json').exists():
            plan = read_json(child/'plan.json')
            rows = plan['rows']
        else:
            rows = []
            for split in ('training', 'selection', 'evaluation'):
                count = cfg['synthetic'][split+'_repeats']
                for repeat in range(count):
                    for record_index in range(cfg['synthetic']['records_per_identity']):
                        row, arrays = synthetic_record(cfg, scenario, split, repeat, record_index=record_index)
                        row['split'] = 'train' if split == 'training' else split
                        path = _prepared_path(child, row['id'])
                        if not path.with_suffix('.json').exists():
                            write_npz(path, **arrays)
                            write_json(path.with_suffix('.json'), dict(status='completed', row=row,
                                provenance='known_truth_generator_separate_split_seed_independent_record'))
                        rows.append(row)
            inventory = dict(rows=rows, excluded=[], subjects={s: sorted({r['subject'] for r in rows if r['split'] == s})
                                                            for s in ('train', 'selection', 'evaluation')})
            plan = plan_manifest(cfg, inventory)
            immutable_json(child/'plan.json', plan)
        fit_coordinate(child, rows, cfg)
        calibration = parallel_stage(_calibrate_job, [(cfg, str(child), rows, arm) for arm in arm_specs(cfg)], workers, child, 'calibrate')
        lookup = {r['id']: r for r in rows}
        evaluation = parallel_stage(_evaluate_job,
            [(cfg, str(child), lookup[spec['record_id']], spec) for spec in plan['cells']], workers, child, 'measured')
        summary = summarize(cfg, child)
        summary.update(scope=cfg['synthetic']['scope'], calibration_oracle=False,
            calibration_scope='within_scenario_disjoint_seed_calibration_not_cross_mechanism_transfer',
            scenario=scenario, coordinate_policy=cfg['synthetic']['coordinate_policy'],
            generator_partitions='separate_training_selection_evaluation_seeds_subjects', stages=[calibration, evaluation],
            identical_sources_interpretation='temporal_rank_collapse_not_K1_readout_null' if scenario == 'identical_sources' else None)
        write_json(child/'summary.json', summary)
        children.append(dict(scenario=scenario, path=str((child/'summary.json').relative_to(out)),
            execution=summary['execution'], planned=summary['planned'], terminal=summary['terminal'],
            status_counts=summary['status_counts']))
        write_json(Path(out)/'synthetic_progress.json', dict(completed_scenarios=len(children),
            planned_scenarios=len(cfg['synthetic']['scenarios']), scenarios=children))
    return summarize_synthetic(cfg, out)


def smoke_coordinate(training_row, training_record):
    """Freeze smoke weighting/scoring SD from one separate training record."""
    if training_row.get('dataset') != 'synthetic' or training_row.get('split') not in ('train', 'training'):
        raise ValueError('smoke coordinate requires an explicitly training synthetic record')
    eeg, hb = np.asarray(training_record['eeg'], float), np.asarray(training_record['hb'], float)
    if (eeg.ndim != 3 or eeg.shape[1:] != (12, 5) or hb.shape != (len(eeg), 4, 2)
            or len(eeg) < 2 or not np.isfinite(eeg).all() or not np.isfinite(hb).all()):
        raise ValueError('finite complete training smoke panel required')
    observed = np.column_stack((eeg.reshape(len(eeg), -1), hb.reshape(len(hb), -1)))
    before_floor = observed.std(axis=0)
    floor = 1e-6
    return dict(eeg_scale=np.ones((12, 5)), hb_scale=1., eeg_shape=(12, 5), hb_shape=(4, 2),
        sd=np.maximum(before_floor, floor), sd_before_floor=before_floor, sd_floor=floor,
        sd_floor_applied=before_floor < floor, training_ids=[training_row['id']],
        training_subjects=[training_row['subject']], training_rows=len(eeg),
        reference_steps=min(20, max(2, round(len(eeg)/6))),
        fit_scope='same_scenario_separate_training_seed_shared_K1_K2',
        sd_reference_scope='complete_training_window_no_evaluation_statistics',
        coordinate_policy='generator_model_coordinates_identity_scaling_training_only_SD')


def synthetic_smoke(cfg, out):
    """Twelve short fits; known truth EEG readouts are explicitly oracle here."""
    started, records = time.monotonic(), []
    for scenario in cfg['synthetic']['scenarios']:
        training_row, training_record = synthetic_record(cfg, scenario, 'training', 0, smoke=True)
        coordinate = smoke_coordinate(training_row, training_record)
        coordinate_path = Path(out)/'scenarios'/scenario/'coordinate.json'
        immutable_json(coordinate_path, coordinate)
        row, record = synthetic_record(cfg, scenario, 'evaluation', 0, smoke=True)
        for k in (1, 2):
            arm = next(a for a in arm_specs(cfg) if a['K'] == k and a['geometry'] == 'true' and a['hb_pairs'] == 4 and a['eeg_channels'] == 12)
            identity = token(scenario+'|'+arm['id'])
            path = Path(out)/'cells'/(identity+'.json')
            if path.exists():
                records.append(read_json(path))
                continue
            result = dict(id=identity, record_id=row['id'], subject=row['subject'], arm=arm['id'], mode='heldout_pair',
                pairing='paired', method='SSM', scenario=scenario, success=False,
                calibration='oracle_signed_band_readout_for_software_smoke_only',
                coordinate_source=str(coordinate_path.relative_to(out)),
                sd_reference_scope=coordinate['sd_reference_scope'])
            try:
                design = spatial_design(cfg, row['patch'], arm)
                bands = np.asarray([[1., -.7], [-.6, 1.], [1.2, .8], [.4, -.8], [-.8, .5]])[:, :k]
                loading = (np.asarray(design['spatial_eeg'])[:, None, :]*bands[None]).reshape(60, k)
                short = deepcopy(cfg)
                short['solver']['max_nfev'] = min(24, cfg['solver']['max_nfev'])
                fit = fit_record(short, record, coordinate, arm, design, loading, 'heldout_pair', 'paired', 1.)
                result.update(fit_metadata(fit))
                result['known_truth_diagnostics'] = truth_diagnostics(fit, record)
                if fit['success']:
                    result.update(score_prediction(fit['hb_prediction'][:, 0], record['hb'][:, 0],
                        coordinate['sd'][60:62], endpoint_mask(cfg, 'heldout_pair', steps=len(record['eeg']))))
                write_npz(path.with_suffix('.npz'), prediction=fit['prediction'], drivers=fit['drivers'],
                    states=fit['states'], truth_drivers=record['truth_drivers'], truth_states=record['truth_states'])
            except Exception as exc:
                result.update(status='failed_exception', success=False, error=str(exc), traceback=traceback.format_exc())
            write_json(path, result)
            records.append(result)
    summary = dict(stage='synthetic-smoke', execution='completed', planned=12, terminal=len(records),
        succeeded=sum(valid_score(r) for r in records), status_counts=status_counts(records),
        calibration_oracle=True, calibration='known_signed_band_readout_no_training_or_selection_claim',
        coordinate_fit_scope='same_scenario_separate_training_seed_shared_K1_K2',
        sd_reference_scope='complete_training_window_no_evaluation_statistics',
        virtual_overrides=dict(steps=40, temporal_modes=6, max_nfev=min(24, cfg['solver']['max_nfev'])),
        elapsed_s=time.monotonic()-started, cells=records, measured_arrays_read=0,
        scientific_verdict='not_assigned', interpretation='software_smoke_negative_fits_retained_not_qualification')
    write_json(Path(out)/'summary.json', summary)
    return summary


def status_counts(rows):
    return {status: sum(r.get('status') == status for r in rows) for status in sorted({r.get('status', 'missing') for r in rows})}


def valid_score(row):
    return (row.get('status') == 'completed' and bool(row.get('success'))
            and row.get('nmse') is not None and np.isfinite(row['nmse']) and row['nmse'] >= 0)


def available_cell(row):
    return row.get('status') not in ('unavailable_wrong_EEG', 'unavailable_preparation', 'missing')


def subject_endpoint(rows, penalty):
    by_subject, success_subject = {}, {}
    for row in rows:
        if not available_cell(row):
            continue
        success = valid_score(row)
        by_subject.setdefault(row['subject'], []).append(float(row['nmse']) if success else penalty)
        if success:
            success_subject.setdefault(row['subject'], []).append(float(row['nmse']))
    all_subject = {s: float(np.mean(values)) for s, values in by_subject.items()}
    successful = {s: float(np.mean(values)) for s, values in success_subject.items()}
    return dict(planned=len(rows), available=sum(available_cell(r) for r in rows),
        unavailable=sum(not available_cell(r) for r in rows), succeeded=sum(valid_score(r) for r in rows),
        planned_subjects=len({r['subject'] for r in rows}),
        independent_subjects=len(all_subject), status_counts=status_counts(rows),
        subject_nmse_with_failure_penalty=all_subject,
        subject_nrmse_with_failure_penalty={s: np.sqrt(v) for s, v in all_subject.items()},
        mean_subject_nrmse_with_failure_penalty=float(np.mean(np.sqrt(list(all_subject.values())))) if all_subject else None,
        success_subset_mean_subject_nrmse=float(np.mean(np.sqrt(list(successful.values())))) if successful else None,
        success_subset_subjects=len(successful), failure_penalty_nmse=penalty,
        pooling='within_subject_mean_NMSE_then_square_root_then_equal_subject_mean')


def paired_bootstrap(left, right, *, repeats=1000, seed=0):
    """Benefit=reference minus candidate after within-subject NMSE pooling."""
    subjects = sorted(set(left) & set(right))
    if not subjects:
        return dict(subjects=0, mean_nrmse_improvement=None, interval_95=None)
    difference = np.asarray([np.sqrt(right[s])-np.sqrt(left[s]) for s in subjects])
    rng = np.random.default_rng(seed)
    draws = difference[rng.integers(0, len(subjects), (repeats, len(subjects)))].mean(axis=1)
    return dict(subjects=len(subjects), subject_ids=subjects, mean_nrmse_improvement=float(difference.mean()),
        interval_95=np.quantile(draws, [.025, .975]), bootstrap_repeats=repeats,
        direction='positive_favors_candidate', unit='subject_all_windows_together',
        uncertainty='descriptive_frozen_selection_not_population_confirmation')


def compare_cells(candidate_rows, reference_rows, cfg):
    """Use common canonical-window availability before any subject aggregation."""
    def key(row):
        return (row.get('dataset'), row['subject'], row['record_id'], row['mode'], row.get('scenario'))
    left = {key(row): row for row in candidate_rows}
    right = {key(row): row for row in reference_rows}
    common = sorted(set(left) & set(right))
    usable = [k for k in common if available_cell(left[k]) and available_cell(right[k])]
    a = subject_endpoint([left[k] for k in usable], cfg['evaluation']['failure_penalty_nmse'])
    b = subject_endpoint([right[k] for k in usable], cfg['evaluation']['failure_penalty_nmse'])
    bootstrap = paired_bootstrap(a['subject_nmse_with_failure_penalty'], b['subject_nmse_with_failure_penalty'],
        repeats=cfg['evaluation']['bootstrap_repeats'], seed=cfg['seed'])
    return dict(planned_common_cells=len(common), common_available_cells=len(usable),
        unavailable_pair_cells=len(common)-len(usable),
        candidate_unmatched_cells=len(set(left)-set(right)), reference_unmatched_cells=len(set(right)-set(left)),
        status='estimated' if usable else 'not_estimable_no_common_available_control',
        numerical_failures_kept_with_penalty=True, unavailable_is_failure_penalty=False,
        **bootstrap)


def summarize(cfg, out):
    if not (Path(out)/'plan.json').exists() and (Path(out)/'synthetic_progress.json').exists():
        return summarize_synthetic(cfg, out)
    plan = read_json(Path(out)/'plan.json')
    rows = []
    for spec in plan['cells']:
        path = Path(out)/'cells'/(spec['id']+'.json')
        rows.append(read_json(path) if path.exists() else dict(spec, status='missing', success=False))
    groups = {}
    for row in rows:
        key = '|'.join((row.get('scenario', 'public_development'), *(row[k] for k in ('arm', 'mode', 'pairing', 'method'))))
        groups.setdefault(key, []).append(row)
    penalty = cfg['evaluation']['failure_penalty_nmse']
    summary_groups = {k: subject_endpoint(v, penalty) for k, v in groups.items()}
    comparisons = []
    pairs = [('K1_true_H4_E12', 'K1_sham_H4_E12', 'true_vs_sham_K1'),
        ('K2_true_H4_E12', 'K2_sham_H4_E12', 'true_vs_sham_K2'),
        ('K1_true_H4_E12', 'K1_uniform_H4_E12', 'true_vs_uniform_K1'),
        ('K2_true_H4_E12', 'K1_true_H4_E12', 'K2_vs_K1'),
        ('K1_true_H4_E12', 'K1_true_H2_E12', 'multi_vs_small_Hb'),
        ('K1_true_H4_E12', 'K1_true_H4_E6', 'multi_vs_small_EEG')]
    for scenario in sorted({r.get('scenario', 'public_development') for r in rows}):
        for candidate, reference, label in pairs:
            for mode in MODES:
                for pairing in PAIRINGS:
                    a, b = (f'{scenario}|{arm}|{mode}|{pairing}|SSM' for arm in (candidate, reference))
                    if a in groups and b in groups:
                        comparisons.append(dict(comparison=label, scenario=scenario, candidate=a, reference=b,
                            **compare_cells(groups[a], groups[b], cfg)))
        for arm in arm_specs(cfg):
            for mode in MODES:
                candidate = f'{scenario}|{arm["id"]}|{mode}|paired|SSM'
                references = [(f'{scenario}|{arm["id"]}|{mode}|{pairing}|SSM', 'paired_vs_'+pairing) for pairing in ('Hb_only', 'wrong_EEG')]
                references += [(f'{scenario}|{arm["id"]}|{mode}|paired|{method}', 'model_vs_'+method) for method in BASELINES]
                for reference, label in references:
                    if candidate in groups and reference in groups:
                        comparisons.append(dict(comparison=label, scenario=scenario, candidate=candidate, reference=reference,
                            same_input_ridge_comparison=mode == 'heldout_pair' if 'ridge' in label else None,
                            **compare_cells(groups[candidate], groups[reference], cfg)))
            if arm['hb_pairs'] == 4 and arm['eeg_channels'] == 12:
                for mode in ('heldout_pair', 'center_pair'):
                    method_pairs = [('SSM', 'SSM_fixed1', 'selected_vs_fixed1')]
                    if arm['K'] == 2:
                        method_pairs += [('SSM_fixed1', 'SSM_tied_fixed1', 'K2_free_vs_tied_fixed1')]
                    for method, reference_method, label in method_pairs:
                        candidate = f'{scenario}|{arm["id"]}|{mode}|paired|{method}'
                        reference = f'{scenario}|{arm["id"]}|{mode}|paired|{reference_method}'
                        if candidate in groups and reference in groups:
                            comparisons.append(dict(comparison=label, scenario=scenario,
                                candidate=candidate, reference=reference,
                                interpretation='increment_of_two_independent_trajectories_and_initials' if reference_method == 'SSM_tied_fixed1' else
                                    'fixed_readout_weight1_diagnostic_not_selection',
                                **compare_cells(groups[candidate], groups[reference], cfg)))
    result = dict(execution='completed' if all(r['status'] != 'missing' for r in rows) else 'incomplete',
        planned=len(rows), terminal=sum(r['status'] != 'missing' for r in rows),
        status_counts=status_counts(rows), groups=summary_groups, comparisons=comparisons,
        mechanisms_pooled=False,
        primary='heldout_pair_HbO_HbR_training_SD_NRMSE', scientific_verdict='not_assigned',
        claim_limits=cfg['claim_limits'], source='immutable_plan_plus_terminal_cell_records')
    write_json(Path(out)/'summary.json', result)
    return result


def summarize_synthetic(cfg, out):
    children, counts = [], {}
    for scenario in cfg['synthetic']['scenarios']:
        path = Path(out)/'scenarios'/scenario/'summary.json'
        if path.exists():
            summary = read_json(path)
            children.append(dict(scenario=scenario, path=str(path.relative_to(out)),
                execution=summary['execution'], planned=summary['planned'], terminal=summary['terminal'],
                status_counts=summary['status_counts']))
        else:
            plan_path = path.parent/'plan.json'
            planned = read_json(plan_path)['planned_cells'] if plan_path.exists() else None
            children.append(dict(scenario=scenario, path=str(path.relative_to(out)), execution='missing',
                planned=planned, terminal=0, status_counts={'missing_summary': 1}))
        for status, count in children[-1]['status_counts'].items():
            counts[status] = counts.get(status, 0)+count
    result = dict(execution='completed' if all(c['execution'] == 'completed' for c in children) else 'incomplete',
        planned=sum(c['planned'] for c in children) if all(c['planned'] is not None for c in children) else None,
        terminal=sum(c['terminal'] for c in children), status_counts=counts, scenarios=children,
        calibration_oracle=False, calibration_scope='within_scenario_disjoint_seed_calibration_not_cross_mechanism_transfer',
        mechanisms_pooled=False, scientific_verdict='not_assigned', scope=cfg['synthetic']['scope'],
        coordinate_policy=cfg['synthetic']['coordinate_policy'], source='scenario_child_summaries')
    write_json(Path(out)/'summary.json', result)
    return result


def immutable_json(path, value):
    if Path(path).exists():
        if read_json(path) != serial(value):
            raise ValueError(f'immutable run identity changed: {path}')
    else:
        write_json(path, value)


def resource_metadata():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    result = dict(cpu_affinity=affinity, logical_cpus=os.cpu_count(), python=platform.python_version(),
        platform=platform.platform(), numerical_threads=1, executable=sys.executable,
        package_versions={name: package_version(name) for name in ('numpy', 'scipy', 'numba', 'PyYAML')})
    for name, filename in (('cpu_quota', '/sys/fs/cgroup/cpu.max'), ('memory_limit', '/sys/fs/cgroup/memory.max')):
        try:
            result[name] = Path(filename).read_text().strip()
        except OSError:
            result[name] = None
    return result


def snapshot_sources(out, config_path):
    """Retain actual uncommitted source, not only the repository HEAD label."""
    names = (
        'experiments/scripts/evaluate_shared_driver_spatial.py',
        'src/inference/shared_driver_spatial.py', 'src/inference/shared_driver_reconstruction.py',
        'src/inference/shared_driver_rk4.py', 'src/inference/shared_driver_modes.py',
        'src/inference/observation_baselines.py', 'src/inference/t3a_balloon_robust_ssm.py',
        'src/data/shared_driver_spatial.py', 'src/data/unified_physiology.py',
        'src/data/clean_physiology_cache.py', 'src/data/event_alignment.py',
        'src/data/homer2_preprocessing.py', 'src/data/registry.py',
    )
    manifest, directory = [], Path(out)/'source_snapshot'
    for name in names:
        source, target = CODE_ROOT/name, directory/name
        if not source.is_file():
            raise ValueError('required source missing: '+name)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('run source changed; use a fresh versioned export: '+name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifest.append(dict(path=name, bytes=source.stat().st_size,
                             retained=str(target.relative_to(out))))
    target = directory/'resolved_input_config.yaml'
    source = Path(config_path)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError('immutable input YAML changed')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    immutable_json(directory/'files.json', manifest)


def supervisor_metadata():
    names = ('SYSTEMD_EXEC_PID', 'INVOCATION_ID', 'SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID')
    return {key: os.environ[key] for key in names if os.environ.get(key)}


def check(cfg):
    params = parameters(cfg)
    arms = arm_specs(cfg)
    for arm in arms:
        design = spatial_design(cfg, _synthetic_patch(), arm)
        if np.linalg.matrix_rank(design['spatial_eeg']) != arm['K']:
            raise ValueError('rank deficient registered EEG design')
    return dict(stage='check', status='completed', arms=arms, measured_arrays_read=0,
        parameters_valid=True, protected_data='forbidden', default_stage='check',
        numerical_backend=cfg['solver']['numerical_backend'])


def dryrun(cfg):
    result = check(cfg)
    rows = []
    for split in ('train', 'selection', 'evaluation'):
        for index in range(2):
            rows.append(dict(id=f'{split}_{index}', dataset='synthetic', subject=split,
                split=split, record=f'{split}_{index}', task='fixture', condition='same',
                eeg_start_s=0., hb_start_s=0., window_seconds=30., patch=_synthetic_patch()))
    plan = plan_manifest(cfg, dict(rows=rows))
    result.update(stage='dryrun', planned_cells=plan['planned_cells'],
        planned_evaluation_windows=2, paired_wrong_EEG_available=all(plan['wrong_eeg_partners'].values()),
        validation='synthetic_metadata_shapes_and_registered_masks_only')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--project-root', type=Path, default=CODE_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=STAGES, default='check')
    parser.add_argument('--workers', type=int, default=1)
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(cli_argv)
    cfg = read_config(args.config)
    if not 1 <= args.workers <= cfg['resources']['max_workers']:
        parser.error('workers must be within the registered 1..24 bound')
    if args.stage == 'check' and args.output is None:
        print(json.dumps(serial(check(cfg)), ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error('an explicit fresh versioned --output is required for this stage')
    root = args.project_root.resolve()
    out = args.output.resolve()
    owning_root = (root/cfg['artifact_root']).resolve()
    if not out.is_relative_to(owning_root) or out == owning_root:
        parser.error('output must be a versioned child of the owning artifact root')
    durable = supervisor_metadata()
    if args.stage in ('synthetic', 'prepare', 'calibrate', 'measured') and not durable:
        parser.error('substantial stages require an existing durable systemd/scheduler supervisor')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another process owns this run')
        immutable_json(out/'resolved_config.json', cfg)
        snapshot_sources(out, args.config)
        existing = read_json(out/'run_identity.json') if (out/'run_identity.json').exists() else None
        family = 'neutral' if args.stage in ('check', 'dryrun', 'summarize') else 'synthetic' if args.stage == 'synthetic' else 'synthetic-smoke' if args.stage == 'synthetic-smoke' else 'public'
        if existing and family != 'neutral' and existing['family'] not in ('neutral', family):
            parser.error('do not mix synthetic/public evidence families in one output')
        if existing and existing['family'] == 'neutral' and family != 'neutral':
            existing['family'] = family
            write_json(out/'run_identity.json', existing)
        if existing is None:
            try:
                sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                sha = None
            immutable_json(out/'run_identity.json', dict(schema=cfg['schema'], family=family,
                created_utc=datetime.now(timezone.utc).isoformat(), source_sha=sha,
                source_worktree_status=subprocess.run(['git', 'status', '--short'], cwd=CODE_ROOT,
                    capture_output=True, text=True).stdout, project_root=str(root)))
        execution = dict(stage=args.stage, command=[sys.executable, str(Path(__file__)), *cli_argv],
            argv=cli_argv, working_directory=str(Path.cwd()),
            started_utc=datetime.now(timezone.utc).isoformat(), status='running',
            supervisor=durable, resources=resource_metadata(), workers=args.workers,
            max_in_flight=args.workers*2, output=str(out))
        execution_path = out/'executions'/(args.stage+'_'+uuid4().hex+'.json')
        execution['history_path'] = str(execution_path.relative_to(out))
        write_json(execution_path, execution)
        write_json(out/'execution.json', execution)
        try:
            if args.stage == 'check':
                result = check(cfg)
            elif args.stage == 'dryrun':
                result = dryrun(cfg)
            elif args.stage == 'synthetic-smoke':
                result = synthetic_smoke(cfg, out)
            elif args.stage == 'synthetic':
                result = synthetic_stage(cfg, out, args.workers)
            elif args.stage == 'plan':
                from src.data.shared_driver_spatial import inventory_spatial
                plan = plan_manifest(cfg, inventory_spatial(cfg, root))
                immutable_json(out/'plan.json', plan)
                result = dict(stage='plan', status=plan['status'], planned_cells=plan['planned_cells'],
                              measured_arrays_read=0, subjects=plan['subjects'])
            elif args.stage == 'prepare':
                plan = read_json(out/'plan.json')
                if plan.get('status') != 'available':
                    raise ValueError('unavailable public plan cannot prepare measured observations')
                result = parallel_stage(_prepare_job,
                    [(cfg, str(root), str(out), row) for row in plan['rows']], args.workers, out, 'prepare')
            elif args.stage == 'calibrate':
                plan = read_json(out/'plan.json')
                if plan.get('status') != 'available':
                    raise ValueError('unavailable plan cannot calibrate')
                fit_coordinate(out, plan['rows'], cfg)
                result = parallel_stage(_calibrate_job,
                    [(cfg, str(out), plan['rows'], arm) for arm in arm_specs(cfg)], args.workers, out, 'calibrate')
            elif args.stage == 'measured':
                plan = read_json(out/'plan.json')
                if plan.get('status') != 'available':
                    raise ValueError('unavailable plan cannot evaluate')
                lookup = {row['id']: row for row in plan['rows']}
                result = parallel_stage(_evaluate_job,
                    [(cfg, str(out), lookup[cell['record_id']], cell) for cell in plan['cells']],
                    args.workers, out, 'measured')
            else:
                result = summarize(cfg, out)
            write_json(out/(args.stage.replace('-', '_')+'_stage.json'), result)
            execution.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            execution.update(status='failed', error=str(exc), traceback=traceback.format_exc(),
                             finished_utc=datetime.now(timezone.utc).isoformat())
            raise
        finally:
            write_json(execution_path, execution)
            write_json(out/'execution.json', execution)
    print(json.dumps(serial({k: v for k, v in result.items() if k not in ('cells', 'groups', 'comparisons')}), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
