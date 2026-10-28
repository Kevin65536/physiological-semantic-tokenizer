"""Versioned public 10 Hz targets for independent S/residual tokenizers.

The parent supplies identities only. Native EEG and the v5/none Hb cache are
read again after checking that identity and split. The teacher is privileged
supervision; student inputs contain only a two-second modality patch. Hb keeps
the declared [-30,-25] s reference and its full-record cache preprocessing.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .functional_tokenizer import inventory, identity_digest
from .ssm_prepared import fit_feature_coordinate, EEG_PROXY_INPUT_SCHEMA
from .unified_physiology import (
    CACHED_FNIRS_INPUT_SCHEMA, load_cached_fnirs_record, load_native_eeg_record,
)
from src.inference.observation_baselines import eeg_band_power, disjoint_native_bin_operator


PARENT = 'experiments/runs/physiology_semantic_tokenizer/functional_tokenizer/20261011_v1'
DATASETS = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
DT = .1
STEPS = 900
STUDENT = slice(300, 500)


def _json(value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=convert, sort_keys=True, ensure_ascii=False)


def _check_scope(project_root, cfg):
    """Resolve the one admitted public parent before any measured array read."""
    root = Path(project_root).resolve()
    data = cfg['data']
    tensor = cfg['tensor']
    parent = (root / cfg['parent_run']).resolve()
    if (parent != (root / PARENT).resolve()
            or tuple(data['datasets']) != DATASETS
            or data['protected_data'] != 'forbidden'
            or data.get('labels_as_training_inputs') is not False
            or data.get('fnirs_input') != CACHED_FNIRS_INPUT_SCHEMA
            or (root / data['cache_root']).resolve() != (root / 'data/cache/physiology_semantic_clean_v5').resolve()
            or cfg['seed'] != 20261010
            or data['train_fraction'] != .6 or data['validation_fraction'] != .2):
        raise ValueError('dual tokenizer requires its exact public parent, v5/none cache and frozen subject split')
    expected = dict(teacher_seconds=90, teacher_steps=900, dt_s=.1,
        student_steps=[300, 500], student_seconds=20, patch_seconds=2,
        patches=10, eeg_rate_hz=200)
    if (any(tensor.get(key) != value for key, value in expected.items())
            or tuple(map(tuple, tensor.get('eeg_bands_hz', ()))) != BANDS):
        raise ValueError('90 s native-recomputed 10 Hz teacher and local two-second input contract required')
    return root, parent


def metadata_inventory(project_root, cfg):
    """Check the full 5,008-trial metadata plan without opening signal arrays."""
    root, parent = _check_scope(project_root, cfg)
    retained = json.loads((parent / 'inventory.json').read_text())
    current = inventory(root, cfg)
    if retained['rows'] != current['rows'] or retained['subjects'] != current['subjects']:
        raise ValueError('current registry identities/geometry/splits differ from frozen parent inventory')
    return dict(current, schema='dual_component_tokenizer_inventory_v1',
        parent_run=str(parent), parent_inventory_schema=retained['schema'],
        planned_trials=len({r['trial_id'] for r in current['rows']}),
        parent_teacher_success_filter=False, measured_arrays_read=0,
        information_scope='local_modality_patch_with_fixed_Hb_reference; processed_coordinate_completion')


def _registered_record(project_root, cfg, rows):
    from .clean_physiology_cache import CleanPhysiologyCacheIndex
    from .registry import REGISTERED_DATASETS
    root, parent = _check_scope(project_root, cfg)
    admitted = {r['id']: r for r in json.loads((parent / 'inventory.json').read_text())['rows']}
    if (not rows or len({r['join_key'] for r in rows}) != 1
            or any(admitted.get(r['id']) != r for r in rows)
            or any(r['split'] not in ('train', 'validation', 'test') for r in rows)):
        raise ValueError('exact parent public identities and subject partitions required before array access')
    index = CleanPhysiologyCacheIndex(root / cfg['data']['cache_root'])
    matches = [r for r in index.records if r.join_key == rows[0]['join_key']]
    if len(matches) != 1:
        raise ValueError('unique central registry record required')
    record = matches[0]
    if any(r['source_identity'] != record.manifest['source_files'] for r in rows):
        raise ValueError('registered source identity changed')
    admitted_root = (root / REGISTERED_DATASETS[record.dataset_id].default_root).resolve()
    for source in record.manifest['source_files']:
        if not (root / source['path']).resolve().is_relative_to(admitted_root):
            raise ValueError('source is outside the registered public dataset root')
    return root, record


def prepare_cached_record(project_root, cfg, rows):
    """Yield (row, arrays, error), reading a native record only once per group."""
    root, record = _registered_record(project_root, cfg, rows)
    # Both identity and the cache contract are checked before either reader.
    from .clean_physiology_cache import require_measurement_producer
    require_measurement_producer(record.manifest)
    bandpass = record.manifest.get('homer2_aligned_contract', {}).get('quality', {}).get('bandpass', {})
    if (record.manifest.get('processing_schema') != 'physiology_measurement_alignment_v5'
            or record.manifest.get('motion_method') != 'none'
            or bandpass.get('status') != 'applied'
            or (bandpass.get('low_hz'), bandpass.get('high_hz')) != (.01, .2)):
        raise ValueError('current v5/none bandpassed measurement cache required before reading EEG')
    native = load_native_eeg_record(root, record)
    cached = load_cached_fnirs_record(root, record)
    if (native.sample_rate_hz != 200 or native.native_unit not in ('uV', 'µV', 'μV')
            or not native.unit_evidence
            or list(native.channel_names) != record.manifest['measurement']['eeg_channel_names']):
        raise ValueError('verified 200 Hz native EEG units and exact channels required')
    channels = {name.upper(): i for i, name in enumerate(native.channel_names)}
    for row in rows:
        try:
            start = round(row['eeg_start_s'] * 200)
            selected = [channels[n.upper()] for n in row['anchor']['eeg_channels']]
            e = np.asarray(native.values[start:start + 18000, selected], float)
            pair = int(row['anchor']['hb_pair'])
            if cached['channel_names'][2 * pair] != row['anchor']['hb_channel'] + '_HbO':
                raise ValueError('registered Hb channel identity mismatch')
            hb_start, hb_end = row['hb_start_s'], row['hb_start_s'] + 90
            # Marker milliseconds and the cache sample grid can represent the
            # same endpoint one ULP apart (108.80000000000001 vs 108.8).
            # Admit only that floating-point roundoff, at both half-open
            # boundaries. Do not enlarge a genuine window or invent support.
            clock_tolerance = 8 * np.spacing(max(abs(hb_start), abs(hb_end), 1.))
            take = ((cached['time_s'] >= hb_start - clock_tolerance)
                & (cached['time_s'] < hb_end - clock_tolerance))
            times = cached['time_s'][take] - hb_start
            times[np.abs(times) <= clock_tolerance] = 0.
            h = cached['values'][take, pair]
            if (start < 0 or e.shape != (18000, 6) or not np.isfinite(e).all()
                    or not np.isfinite(h).all()):
                raise ValueError('complete finite native EEG and supported cached Hb required')
            power = eeg_band_power(e, bands=BANDS, sample_rate=200, target_rate=10)
            if power.shape != (STEPS, 6, 5) or np.any(power <= 0) or not np.isfinite(power).all():
                raise ValueError('native-recomputed 10 Hz five-band power required')
            features = np.log(power).reshape(STEPS, 30)
            features -= features[:50].mean(axis=0)
            # Means use disjoint cached 100 ms bins, then the fixed reference.
            groups = np.floor(np.maximum(times, 0.) / DT + 1e-10).astype(int)
            counts = np.bincount(groups, minlength=STEPS)
            if len(counts) != STEPS or np.any(counts == 0):
                raise ValueError('every 100 ms bin requires recorded cached Hb support')
            hb = np.zeros((STEPS, 2), float)
            np.add.at(hb, groups, h)
            hb /= counts[:, None]
            hb -= hb[:50].mean(axis=0)
            provenance = dict(canonical_identity=row, native_EEG_source=str(native.source_path),
                eeg_channels=row['anchor']['eeg_channels'], eeg_unit=native.native_unit,
                eeg_unit_evidence=native.unit_evidence, eeg_rate_hz=200,
                native_start_sample=start, start_rounding_s=start / 200 - row['eeg_start_s'],
                teacher_EEG_filter_support_event_s=[-30, 60], teacher_proxy_rate_hz=10,
                teacher_bands_hz=BANDS, teacher_reference_event_s=[-30, -25],
                student_EEG_support_event_s=[0, 20], student_patch_seconds=2,
                student_Hb_reference_event_s=[-30, -25], hb=cached['provenance'],
                hb_clock_roundoff_tolerance_s=clock_tolerance,
                inference='offline_processed_coordinate_completion_not_raw_sample_isolation')
            payload = dict(eeg=e[6000:10000].astype('float32'), features=features, hb=hb,
                native_time=times, time_s=np.arange(STEPS) * DT,
                eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA), eeg_proxy_rate_hz=np.array(10.),
                fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA),
                identity=np.array(row['id']), trial_id=np.array(row['trial_id']),
                provenance=np.array(_json(provenance)))
            yield row, payload, None
        except (ValueError, KeyError, IndexError, FloatingPointError) as exc:
            yield row, None, str(exc)


def _check_prepared(arrays):
    if (str(np.asarray(arrays.get('eeg_proxy_input', ''))) != EEG_PROXY_INPUT_SCHEMA
            or float(np.asarray(arrays.get('eeg_proxy_rate_hz', np.nan))) != 10.
            or str(np.asarray(arrays.get('fnirs_input', ''))) != CACHED_FNIRS_INPUT_SCHEMA
            or np.shape(arrays.get('features')) != (STEPS, 30)
            or np.shape(arrays.get('hb')) != (STEPS, 2)
            or np.shape(arrays.get('eeg')) != (4000, 6)
            or not np.array_equal(arrays.get('time_s'), np.arange(STEPS) * DT)):
        raise ValueError('versioned 90 s 10 Hz prepared features and 20 s native EEG required')
    if any(not np.isfinite(arrays[k]).all() for k in ('features', 'hb', 'eeg')):
        raise ValueError('finite complete prepared support required')


def fit_training_coordinates(rows, read_arrays):
    """Fit region/dataset PCA, common Hb scale and native-input SD on train only."""
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'train': groups[row['coordinate']].append(row)
    coordinates = {}
    for key, group in groups.items():
        chunks = []
        count, total, squared = 0, 0., 0.
        for row in group:
            a = read_arrays(row)
            _check_prepared(a)
            chunks.append((a['features'], a['hb']))
            e = np.asarray(a['eeg'], float)
            total += float(e.sum()); squared += float(np.square(e).sum()); count += e.size
        co = fit_feature_coordinate(chunks)
        co.update(eeg_mean=total / count,
            eeg_scale=max(float(np.sqrt(max(squared / count - (total / count) ** 2, 0))), 1e-8),
            training_ids=[r['id'] for r in group],
            training_subjects=sorted({r['subject'] for r in group}),
            eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA, eeg_proxy_rate_hz=10., dt_s=DT,
            fnirs_input=CACHED_FNIRS_INPUT_SCHEMA)
        coordinates[key] = co
    if {r['coordinate'] for r in rows} != set(coordinates):
        raise ValueError('each evaluation coordinate requires training-subject support')
    return coordinates


def produce_teacher(arrays, coordinate, tak_cfg, *, arm_name='tak_fixed', warm_start=None):
    """Fit the declared Tak arm; return observation-space u, S and full residual.

    Fixed-reference Tak is the default engineering teacher. ``tak_fit`` may be
    requested explicitly in the versioned run, retaining per-window privileged
    kappa/tau inference. Neither variant is physiological ground truth.
    """
    from src.inference.shared_driver_reconstruction import (
        fit_nonlinear_shared_parameters, nonlinear_driver_forward,
    )
    from src.inference.t3a_balloon_robust_ssm import (
        BalloonParameters, BalloonFixedParameters, BalloonFreeParameters,
    )
    _check_prepared(arrays)
    if (tak_cfg.get('schema') != 'shared_driver_tak_10hz_v2' or tak_cfg['tensor']['dt_s'] != DT
            or coordinate.get('eeg_proxy_input') != EEG_PROXY_INPUT_SCHEMA
            or arm_name not in ('tak_fixed', 'tak_fit')):
        raise ValueError('current Tak 10 Hz target contract and matching train coordinate required')
    arm = next(a for a in tak_cfg['arms'] if a['name'] == arm_name)
    if arm['hemodynamics'] != 'tak' or arm['initial'] != 'volume_tied' or arm['tau_v'] != 2:
        raise ValueError('Tak with four vascular initial coordinates and fixed tau_v=2 required')
    p, solver, fitting = tak_cfg['parameters'], tak_cfg['solver'], tak_cfg['fitting']
    parameters = BalloonParameters(
        fixed=BalloonFixedParameters(**{k: p[k] for k in
            ('gamma', 'alpha', 'E0', 'P0', 'Q0', 'neurovascular_gain')}),
        free=BalloonFreeParameters(tau=p['tau'], kappa=p['kappa']))
    u = np.column_stack((arrays['features'] @ np.asarray(coordinate['pc']) * coordinate['eeg_factor'],
        arrays['hb'] * coordinate['hb_factor']))
    ops = disjoint_native_bin_operator(arrays['native_time'], steps=STEPS, dt=DT, reference_steps=50)
    operator = np.zeros((3 * STEPS, 3 * STEPS))
    for k, op in enumerate((ops['eeg'], ops['model'], ops['model'])):
        operator[k::3, k::3] = op
    mask = np.ones_like(u, bool); mask[-1] = False
    names = arm['fit']
    values = [fitting['reference'][name] for name in names]
    starts = [dict(parameter_values=np.asarray(values))]
    if names:
        starts = [dict(parameter_values=np.asarray(v)) for v in fitting['starts_kappa_tau']]
    if warm_start is not None:
        if (np.shape(warm_start.get('driver')) != (STEPS,)
                or np.shape(warm_start.get('initial_state')) not in ((4,), (5,))):
            raise ValueError('one whole-trajectory driver and physical initial state required')
        for start in starts:
            start.update(driver=np.asarray(warm_start['driver'])[None],
                initial_state=np.asarray(warm_start['initial_state'])[None])
    kwargs = dict(parameter_names=names, parameter_bounds=[fitting['bounds'][n] for n in names],
        sd=np.asarray(coordinate['sd']), mean_operator=operator, visible=mask[None], starts=starts,
        penalty=solver['penalty'], initial_penalty=solver['initial_penalty'],
        flow_prior_weight=solver['flow_prior_weight'], flow_prior_log_sd=solver['flow_prior_log_sd'],
        substeps=solver['substeps'], max_iterations=solver['max_iterations'],
        max_evaluations_per_trial=solver['max_evaluations_per_trial'],
        gradient_tolerance=solver['gradient_tolerance'], numerical_backend='numba',
        hemodynamics='tak', viscoelastic_s=2., initial_state_mode='volume_tied',
        initial_coordinate_weights=solver['initial_weights_tied'],
        data_weight=DT / solver['observation_reference_dt_s'])
    if names:
        kwargs.update(parameter_prior_log_mean=np.log(values),
            parameter_prior_precision=1 / np.square(arm['log_sd']))
    fit = fit_nonlinear_shared_parameters(u[None], parameters, DT, **kwargs)
    status = {k: fit.get(k) for k in ('status', 'converged', 'objective', 'evaluations',
        'projected_scaled_gradient_inf_norm', 'objective_components', 'parameter_names',
        'parameter_values', 'kappa', 'tau', 'gamma', 'viscoelastic_s')}
    status.update(teacher_arm=arm_name, interpretation='privileged_conditional_model_output_not_physiological_truth')
    if 'prediction' not in fit:
        status['accepted'] = False
        return None, status
    physical = fit['prediction'][0]
    refined_parameters = BalloonParameters(fixed=parameters.fixed,
        free=BalloonFreeParameters(tau=fit['tau'], kappa=fit['kappa']))
    fine = nonlinear_driver_forward(fit['driver'][0], fit['initial_state'][0], refined_parameters, DT,
        substeps=solver['validation_substeps'], derivative=False, numerical_backend='numba',
        hemodynamics='tak', viscoelastic_s=2.)
    refined = (operator @ fine['canonical_prediction'].ravel()).reshape(u.shape)
    integration_error = float(np.max(np.abs((refined - physical)[:-1] / coordinate['sd'])))
    status.update(integration_max_training_sd=integration_error,
        accepted=bool(fit['converged'] and integration_error <= solver['maximum_integration_difference_training_sd']))
    payload = dict(u=u, physical=physical, remaining=u - physical, visible=mask,
        states=fit['states'][0], driver=fit['driver'][0], initial_state=fit['initial_state'][0],
        time_s=np.arange(STEPS) * DT,
        identity=np.asarray(arrays['identity']), teacher_arm=np.array(arm_name),
        objective=np.array(fit['objective']),
        eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA), fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA))
    if not np.allclose(payload['u'], payload['physical'] + payload['remaining'], rtol=0, atol=1e-12):
        raise AssertionError('teacher observation = physical + full residual identity failed')
    return payload, status


def produce_alternative_teacher(arrays, coordinate, tak_cfg, reference, settings):
    """Search the fixed small set of legal whole-trajectory Tak restarts.

    Offsets apply to the four initial coordinates [s, log(f), log(v), log(q)];
    p is derived from v. The first coordinate is additive in the declared s
    units. Driver and all four initial coordinates are then jointly refitted.
    A witness is one distinct converged complete solution, never timewise mixing.
    """
    if str(np.asarray(reference.get('teacher_arm', ''))) != 'tak_fixed':
        raise ValueError('alternative target requires the same fixed-reference Tak teacher')
    initial = np.asarray(reference['initial_state'], float)
    if initial.shape != (5,) or initial[2] != initial[3] or np.any(initial[1:] <= 0):
        raise ValueError('legal four-coordinate Tak initial state required')
    ref_objective = float(np.asarray(reference['objective']))
    maximum = float(settings['maximum_objective_fraction'])
    if maximum < 0 or not np.isfinite(ref_objective):
        raise ValueError('finite reference objective and nonnegative feasibility tolerance required')
    witnesses, eligible = [], []
    for offset in settings['offsets']:
        shifted = initial.copy()
        shifted[0] += float(offset)
        shifted[1:] *= np.exp(float(offset))
        shifted[3] = shifted[2]
        record = dict(offset=float(offset), transformed_coordinates=['s', 'log_f', 'log_v', 'log_q'])
        try:
            candidate, status = produce_teacher(arrays, coordinate, tak_cfg,
                warm_start=dict(driver=np.asarray(reference['driver']), initial_state=shifted))
            record.update(status)
            if candidate is not None:
                fraction = (float(candidate['objective']) - ref_objective) / max(abs(ref_objective), 1e-12)
                difference = float(np.sqrt(np.mean(np.square(
                    (candidate['physical'] - reference['physical']) / np.asarray(coordinate['sd'])))))
                distinct = difference > 1e-5
                record.update(objective_fraction=fraction, physical_RMS_difference_training_sd=difference,
                    distinct=distinct, eligible=bool(status['accepted'] and fraction <= maximum and distinct))
                if record['eligible']: eligible.append((float(candidate['objective']), candidate, record))
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            record.update(accepted=False, eligible=False, status='failed_restart', reason=str(exc))
        witnesses.append(record)
    status = dict(status='unavailable', accepted=False, witnesses=witnesses,
        interpretation='finite_same_model_whole_trajectory_restart_witness_not_posterior_or_truth')
    if not eligible: return None, status
    _, chosen, chosen_status = min(eligible, key=lambda item: item[0])
    status.update(status='completed', accepted=True, selected_offset=chosen_status['offset'],
        objective_fraction=chosen_status['objective_fraction'],
        physical_RMS_difference_training_sd=chosen_status['physical_RMS_difference_training_sd'])
    return chosen, status


def synthetic_teacher_check(tak_cfg, seed=20261010):
    """One actual 90 s fixed-Tak fit, without measured files or model mocks."""
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    from src.inference.t3a_balloon_robust_ssm import BalloonParameters
    started = time.monotonic()
    t = np.arange(STEPS) * DT
    rng = np.random.default_rng(seed)
    driver = .015 * np.sin(2 * np.pi * t / 17) + .008 * np.cos(2 * np.pi * t / 7)
    forward = nonlinear_driver_forward(driver, np.r_[0., np.ones(4)], BalloonParameters(), DT,
        substeps=8, derivative=False, numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
    u = forward['canonical_prediction'].copy()
    u -= u[:50].mean(0)
    arrays = dict(eeg=rng.normal(size=(4000, 6)).astype('float32'),
        features=np.pad(u[:, 0, None], ((0, 0), (0, 29))), hb=u[:, 1:],
        native_time=t, time_s=t, eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),
        eeg_proxy_rate_hz=np.array(10.), fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA),
        identity=np.array('synthetic_fixed_Tak_90s'))
    co = dict(pc=np.r_[1., np.zeros(29)], eeg_factor=1., hb_factor=1.,
        sd=np.array([.025, .025, .025]), eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA)
    targets, status = produce_teacher(arrays, co, tak_cfg)
    if targets is None or not status['accepted']:
        raise AssertionError('actual synthetic fixed-Tak teacher did not pass numerical acceptance')
    closure = float(np.max(np.abs(targets['u'] - targets['physical'] - targets['remaining'])))
    if closure > 1e-12: raise AssertionError('synthetic teacher decomposition did not close')
    return dict(status='completed', measured_arrays_read=0, steps=STEPS, dt_s=DT,
        teacher_status=status, decomposition_max_error=closure,
        physical_NRMSE_training_sd=np.sqrt(np.mean(np.square(
            (targets['physical'] - targets['u']) / co['sd']), axis=0)),
        elapsed_seconds=time.monotonic() - started,
        implementation='actual_numba_Tak_four_and_eight_substeps_with_fixed_reference_parameters')


def pack_training_cache(cfg, rows, coordinates, read_prepared, read_teacher, output):
    """Write mmap-friendly patch arrays on complete matched three-region trials.

    Readers receive one metadata row. A missing/failed teacher returns None or
    raises FileNotFoundError. The runner must finish all teacher task records
    before calling this function; no success-subset selection occurs here.
    """
    bytrial = defaultdict(list)
    for row in rows: bytrial[row['trial_id']].append(row)
    retained, failures = [], []
    for trial_id, group in sorted(bytrial.items()):
        success = True
        for row in group:
            try: teacher = read_teacher(row)
            except FileNotFoundError: teacher = None
            if teacher is None:
                success = False
        if len(group) != 3 or not success:
            failures.append(trial_id)
        else:
            retained.extend(group)
    if not retained: raise ValueError('no complete matched Tak-teacher trial population')
    # Observation SD is fit on retained training subjects, shared by both
    # component targets. No per-patch normalization or target centering.
    moments = {key: [0, np.zeros(3), np.zeros(3)] for key in coordinates}
    for row in retained:
        if row['split'] != 'train': continue
        u = np.asarray(read_teacher(row)['u'][STUDENT], float)
        n, total, squared = moments[row['coordinate']]
        moments[row['coordinate']] = [n + len(u), total + u.sum(0), squared + np.square(u).sum(0)]
    scales = {}
    for key, (n, total, squared) in moments.items():
        if not n: raise ValueError('retained training support required for every coordinate')
        scales[key] = np.maximum(np.sqrt(np.maximum(squared / n - (total / n) ** 2, 0)), 1e-8)
    shapes = dict(eeg_input=(len(retained), 10, 2400), hb_input=(len(retained), 10, 40))
    for modality, dimension in (('eeg', 20), ('hb', 40)):
        for component in ('u', 's', 'e'): shapes[modality + '_' + component] = (len(retained), 10, dimension)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    maps = {k: np.lib.format.open_memmap(out / (k + '.tmp.npy'), mode='w+', dtype='float32', shape=v)
        for k, v in shapes.items()}
    for i, row in enumerate(retained):
        a, teacher = read_prepared(row), read_teacher(row)
        _check_prepared(a)
        co, scale = coordinates[row['coordinate']], scales[row['coordinate']]
        if str(np.asarray(a['identity'])) != row['id'] or str(np.asarray(teacher['identity'])) != row['id']:
            raise ValueError('canonical prepared/teacher identity mismatch')
        u, s, e = (np.asarray(teacher[k][STUDENT], float) for k in ('u', 'physical', 'remaining'))
        if (u.shape != (200, 3) or not all(np.isfinite(v).all() for v in (u, s, e))
                or not np.allclose(u, s + e, rtol=0, atol=1e-10)):
            raise ValueError('finite same-coordinate teacher decomposition required')
        maps['eeg_input'][i] = ((a['eeg'] - co['eeg_mean']) / co['eeg_scale']).reshape(10, 2400)
        maps['hb_input'][i] = (a['hb'][STUDENT] * co['hb_factor'] / scale[1:]).reshape(10, 40)
        for key, value in (('u', u), ('s', s), ('e', e)):
            scaled = value / scale
            maps['eeg_' + key][i] = scaled[:, 0].reshape(10, 20)
            maps['hb_' + key][i] = scaled[:, 1:].reshape(10, 40)
    for key, value in maps.items():
        value.flush()
        (out / (key + '.tmp.npy')).replace(out / (key + '.npy'))
    manifest = dict(schema='dual_component_training_cache_v1', rows=retained,
        identity=identity_digest(retained), shapes=shapes,
        arrays={key: key + '.npy' for key in shapes}, dtype='float32',
        planned_windows=len(rows), retained_windows=len(retained),
        planned_trials=len(bytrial), retained_trials=len(retained) // 3, failed_trials=failures,
        observation_training_sd=scales, scale_fit='retained_training_subjects_only_no_target_centering',
        target_scaling='divide_by_frozen_observation_training_sd_no_centering',
        target_units='training_observation_SD',
        target_coordinates=['EEG_training_PCA_log_bandpower', 'HbO', 'HbR'],
        decomposition='observation = physical_Tak + full_fixed_teacher_remaining',
        dt_s=DT, student_seconds=20, patch_seconds=2, patches=10,
        teacher_seconds=90, teacher_event_support_s=[-30, 60],
        eeg_input='native_200Hz_voltage_local_2s_no_teacher_features',
        hb_input='cached_v5_none_local_2s_with_fixed_reference_event_minus30_to_minus25',
        hb_preprocessing='full_record_offline_0.01_0.2Hz_before_masking',
        masking_claim='processed_coordinate_completion', labels_as_training_inputs=False,
        parent_run=cfg['parent_run'], frozen_subject_split=True,
        independent_subjects={ds: {split: len({r['subject'] for r in retained
            if r['dataset'] == ds and r['split'] == split})
            for split in ('train', 'validation', 'test')} for ds in DATASETS},
        failures='all branches share complete three-region trial support; all planned failures retained')
    path = out / 'manifest.json'; tmp = path.with_suffix('.tmp')
    tmp.write_text(_json(manifest) + '\n'); tmp.replace(path)
    return manifest
