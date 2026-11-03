"""Native EOG activity anchors on retained Tak windows, without refitting.

Only the central native EEG reader supplies EOG. Scalp EEG is never used as an
EOG substitute. These unfiltered auxiliary potentials can also contain motion,
muscle and contact noise; their amplitude does not identify a physiological
source. The optional counterfactual reuses the existing EOG-only regression;
its nuisance fit is an explicitly offline, full-record sensitivity diagnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .clean_physiology_cache import CleanPhysiologyCacheIndex, DEFAULT_CLEAN_CACHE_ROOT
from .tak_mismatch import (
    BANDS_HZ, _registered_sources, continuous_hb_support_segments, continuous_tak_support,
)
from .ssm_prepared import EEG_PROXY_INPUT_SCHEMA
from .unified_physiology import RAW_DATASET_IDS, load_native_eeg_record
from .eeg_artifact_preprocessing import (
    EEGArtifactCleaningConfig, _robust_eog_regression, detect_ocular_mask,
)
from src.inference.observation_baselines import eeg_band_power


SCHEMA = 'tak_mismatch_native_ocular_v1'
EOG_CHANNELS = ('HEOG', 'VEOG')
SCORE_INTERVAL_S = (5., 29.9)


def native_ocular_window(values, sample_rate_hz):
    """Describe one complete native 30 s EOG window in its original units.

    Inputs are [native sample, HEOG/VEOG]. The score interval is [5,29.9) s,
    matching 10 Hz Tak bins 50..298. RMS and peak subtract each channel's mean
    on this interval. Derivatives are adjacent *native* differences per second
    within that interval. A scalar requires complete channel support; its real
    finite duration is still reported when incomplete. Bin features use the
    same score mean, never an independently fitted bin scale or baseline.
    """
    x = np.asarray(values, dtype=float)
    rate = float(sample_rate_hz)
    if (not np.isfinite(rate) or rate <= 0
            or not np.isclose(rate / 10., round(rate / 10.))
            or x.shape != (round(30 * rate), 2)):
        raise ValueError('complete 30 s native EOG [samples,HEOG/VEOG] on a 10 Hz divisible clock required')
    samples_per_bin = round(rate / 10.)
    start, stop = (round(t * rate) for t in SCORE_INTERVAL_S)
    score = x[start:stop]
    finite = np.isfinite(score)
    complete = finite.all(axis=0)
    center = np.full(2, np.nan)
    center[complete] = score[:, complete].mean(axis=0)
    centered = x - center
    bins = centered.reshape(300, samples_per_bin, 2)
    valid = np.isfinite(x).reshape(300, samples_per_bin, 2).all(axis=1) & complete
    mean = np.full((300, 2), np.nan)
    rms_bins = mean.copy()
    rms = np.full(2, np.nan)
    peak = rms.copy()
    derivative = rms.copy()
    if complete.any():
        with np.errstate(invalid='ignore'):
            bin_mean = bins[:, :, complete].mean(axis=1)
            bin_rms = np.sqrt(np.mean(bins[:, :, complete] ** 2, axis=1))
        mean[:, complete] = np.where(valid[:, complete], bin_mean, np.nan)
        rms_bins[:, complete] = np.where(valid[:, complete], bin_rms, np.nan)
        centered_score = centered[start:stop, complete]
        rms[complete] = np.sqrt(np.mean(centered_score ** 2, axis=0))
        peak[complete] = np.max(np.abs(centered_score), axis=0)
        derivative[complete] = np.sqrt(np.mean((np.diff(score[:, complete], axis=0) * rate) ** 2, axis=0))
    return dict(rms=rms, peak_abs=peak, derivative_rms=derivative,
        rms_combined=float(np.sqrt(np.mean(rms ** 2))) if complete.all() else np.nan,
        valid_seconds=finite.sum(axis=0) / rate, complete=complete,
        score_mean=center, rms_10hz=rms_bins, mean_10hz=mean, valid_mask=valid)


def _ocular_record_context(project_root, record_plan, cache_root, record, index):
    """One identity and clock check shared by both native EOG consumers."""
    root = Path(project_root)
    if index is None:
        index = CleanPhysiologyCacheIndex(root / cache_root)
    matches = [r for r in index.records_by_join_key.get(record_plan['join_key'], [])
               if r.record_id == record_plan['cache_record_id']]
    if len(matches) != 1 or (record is not None and record != matches[0]):
        raise ValueError('frozen canonical cache record unavailable or changed')
    record = matches[0]
    if (record.dataset_id not in RAW_DATASET_IDS or record_plan['dataset'] != record.dataset_id
            or record_plan['subject'] != record.canonical_subject_id
            or record_plan['record'] != record.base_record_id
            or record_plan['source_identity'] != record.manifest['source_files']
            or record_plan['array_shapes'] != record.manifest['array_shapes']):
        raise ValueError('frozen source or canonical identity changed before EOG access')
    sources = _registered_sources(record, root)
    support = continuous_tak_support(record, index.events_by_join_key.get(record.join_key, []),
                                     index.reports_by_join_key.get(record.join_key, []))
    windows = support.pop('windows')
    if (windows != record_plan['windows'] or support != record_plan['coverage']
            or continuous_hb_support_segments(record, index.events_by_join_key.get(record.join_key, []))
            != record_plan.get('continuous_hb_segments')):
        raise ValueError('frozen native clock support changed before EOG access')
    state = record.manifest.get('measurement', {}).get('eeg_preprocessing_state', {})
    return root, record, windows, sources, state


def _require_native_identity(eeg, record_plan, sources, state):
    if (str(Path(eeg.source_path).resolve()) not in sources
            or list(eeg.channel_names) != record_plan['eeg_channel_names']
            or eeg.sample_rate_hz != state.get('native_sample_rate_hz')
            or list(eeg.auxiliary_channel_names) != state.get('auxiliary_channel_names')
            or eeg.native_unit != state.get('native_unit', eeg.native_unit)):
        raise ValueError('native EOG source, scalp identity or clock differs from registration')


def prepare_tak_ocular_record(project_root, record_plan, *,
                              cache_root=DEFAULT_CLEAN_CACHE_ROOT, record=None, index=None):
    """Read native EOG once per registered record, retaining every planned row.

    The parent identity and native clock support are checked before any signal
    read. The caller owns run persistence and scheduling. Records with no
    registered EOG return explicit missing arrays without opening their EEG.
    The output window identity is the parent's native window identity, before
    any Hb-pair expansion, so a single EOG trace must not count as several
    independent observations merely because several Hb pairs share it.
    """
    root, record, windows, sources, state = _ocular_record_context(
        project_root, record_plan, cache_root, record, index)
    count = len(windows)
    output = {k: np.full((count, 2), np.nan)
              for k in ('rms', 'peak_abs', 'derivative_rms', 'valid_seconds', 'score_mean')}
    output.update(rms_combined=np.full(count, np.nan),
        complete=np.zeros((count, 2), bool), rms_10hz=np.full((count, 300, 2), np.nan),
        mean_10hz=np.full((count, 300, 2), np.nan), valid_mask=np.zeros((count, 300, 2), bool),
        window_ids=np.asarray([w['id'] for w in windows]), time_s=np.arange(300) / 10.,
        native_target_sample_bounds=np.full((count, 2), -1, dtype=np.int64),
        native_score_sample_bounds=np.full((count, 2), -1, dtype=np.int64),
        target_start_rounding_s=np.full(count, np.nan))
    metadata = dict(schema=SCHEMA, record_id=record.join_key, dataset=record.dataset_id,
        subject=record.canonical_subject_id, native_record=record.base_record_id,
        source_identity=record_plan['source_identity'], channels=list(EOG_CHANNELS),
        registered_auxiliary_channels=state.get('auxiliary_channel_names', []),
        registered_cache_EEG_branch=state.get('signal_branch'),
        processing='native_unfiltered_EOG; score_interval_channel_mean_removed; no_scaling_or_EOG_regression',
        score_interval_s=list(SCORE_INTERVAL_S), score_interval='left_closed_right_open',
        target_clock='same_native_target_rounding_as_retained_Tak_EEG; no_Hb_clock_repair',
        derivative_unit='native_potential_unit_per_second',
        interpretation='auxiliary_activity_association; not_blink_labels_or_source_identification',
        status='unavailable', reason=None, windows=[dict(window_id=w['id'], status='unavailable',
            reason=w.get('reason'), hb_start_s=w['hb_start_s'], eeg_start_s=w['eeg_start_s'],
            task=w.get('task'), condition=w.get('condition')) for w in windows],
        measured_signal_reads=0, planned_windows=count, available_windows=0)
    output['metadata'] = metadata
    if not state.get('auxiliary_channel_names'):
        metadata['reason'] = 'no_registered_native_EOG_auxiliary'
        for row in metadata['windows']:
            row['reason'] = metadata['reason']
        return output
    if not any(w['status'] == 'available' for w in windows):
        metadata['reason'] = 'no_admitted_paired_native_clock_windows'
        return output
    eeg = load_native_eeg_record(root, record)
    metadata['measured_signal_reads'] = 1
    _require_native_identity(eeg, record_plan, sources, state)
    names = [str(n).upper() for n in eeg.auxiliary_channel_names]
    if (eeg.auxiliary_values is None or any(names.count(c) != 1 for c in EOG_CHANNELS)
            or np.shape(eeg.auxiliary_values) != (len(eeg.values), len(names))):
        metadata['reason'] = 'native_HEOG_VEOG_pair_missing_or_ambiguous'
        return output
    rate = float(eeg.sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0 or not np.isclose(rate / 10., round(rate / 10.)):
        raise ValueError('native EOG clock must be divisible into complete 100 ms bins')
    aux = np.asarray(eeg.auxiliary_values)
    selected = [names.index(c) for c in EOG_CHANNELS]
    metadata.update(native_unit=eeg.native_unit, unit_evidence=eeg.unit_evidence,
        native_sample_rate_hz=rate, source=str(eeg.source_path), reference=eeg.reference,
        native_samples=len(aux), native_auxiliary_channels=list(eeg.auxiliary_channel_names))
    for i, window in enumerate(windows):
        row = metadata['windows'][i]
        if window['status'] != 'available':
            continue
        context_start = round(window['eeg_filter_start_s'] * rate)
        crop_bins = round((window['eeg_start_s'] - window['eeg_filter_start_s']) * 10.)
        start = context_start + crop_bins * round(rate / 10.)
        stop = start + round(30. * rate)
        output['native_target_sample_bounds'][i] = (start, stop)
        output['native_score_sample_bounds'][i] = [start + round(t * rate) for t in SCORE_INTERVAL_S]
        output['target_start_rounding_s'][i] = start / rate - window['eeg_start_s']
        if start < 0 or stop > len(aux):
            row['reason'] = 'incomplete_native_target_support'
            continue
        described = native_ocular_window(aux[start:stop, selected], rate)
        for key, value in described.items():
            output[key][i] = value
        row.update(status='available' if described['complete'].all() else 'incomplete',
            reason=None if described['complete'].all() else 'nonfinite_native_EOG_score_support',
            actual_eeg_start_s=start / rate, actual_score_start_s=start / rate + SCORE_INTERVAL_S[0],
            actual_score_end_s=start / rate + SCORE_INTERVAL_S[1])
    available = int(output['complete'].all(axis=1).sum())
    metadata.update(status='available' if available else 'unavailable',
        reason=None if available else 'no_complete_native_EOG_score_windows',
        planned_windows=count, available_windows=available)
    return output


def amplitude_matched_proxy(raw, clean):
    """Match clean mean/SD using raw shape on score bins 50..298, without Hb."""
    raw, clean = np.asarray(raw, float), np.asarray(clean, float)
    if raw.shape != (300,) or clean.shape != raw.shape or not np.isfinite([raw, clean]).all():
        raise ValueError('two complete 300-bin proxies required')
    r, c = raw[50:299], clean[50:299]
    raw_sd, clean_sd = float(np.std(r)), float(np.std(c))
    state = dict(raw_mean=float(r.mean()), clean_mean=float(c.mean()),
        raw_sd=raw_sd, clean_sd=clean_sd, score_count=len(r), ddof=0,
        clean_constant=clean_sd == 0.)
    if raw_sd <= np.finfo(float).eps * max(1., float(np.max(np.abs(r)))):
        return np.full(300, np.nan), dict(state, status='unavailable_raw_zero_scale')
    matched = c.mean() + (clean_sd / raw_sd) * (raw-r.mean())
    return matched, dict(state, status='available')


def _context_logpower(voltage, window, rate):
    """Use the retained Tak filter context, crop, log and first-5s reference."""
    a, b = [round(window[key]*rate) for key in ('eeg_filter_start_s','eeg_filter_end_s')]
    if a < 0 or b > len(voltage) or b <= a:
        raise ValueError('incomplete native filtering context')
    segment = np.asarray(voltage[a:b], float)
    finite = np.isfinite(segment).all(axis=0)
    power = np.full((round(window['actual_context_s']*10), segment.shape[1], 5), np.nan)
    if finite.any():
        power[:,finite] = eeg_band_power(segment[:,finite], bands=BANDS_HZ,
            sample_rate=rate, target_rate=10.)
    crop = round((window['eeg_start_s']-window['eeg_filter_start_s'])*10)
    power = power[crop:crop+300]
    if power.shape != (300, segment.shape[1], 5):
        raise ValueError('30 s target outside its declared EEG context')
    good = np.isfinite(power).all(axis=(0,2)) & (power>0).all(axis=(0,2))
    features = np.full_like(power,np.nan)
    logpower = np.log(power[:,good])
    features[:,good] = logpower-logpower[:50].mean(axis=0)
    return features


def prepare_tak_ocular_counterfactual(project_root, record_plan, coordinate_models, selected_pairs,
                                     *, prepared, cache_root=DEFAULT_CLEAN_CACHE_ROOT,
                                     record=None, index=None):
    """Apply one offline EOG-only regression, preserving the frozen Tak map.

    ``coordinate_models`` maps each selected pair ID to its existing outer-fold
    coordinate dictionary (``pc['nearest6']`` and ``eeg_factor``). ``prepared``
    is the retained parent record NPZ mapping. No Hb array or Hb fit is read.
    The caller scores raw/clean/amplitude proxies against the same retained
    successful Hb-only readout, using the common support and training SD.

    Native EEG is copied before the existing in-place nuisance regression.
    Its EOG design and robust weights use the complete native record, including
    outside scored windows. This is a processing-sensitivity experiment, not
    independently trained denoising, a new Tak fit or a causal source label.
    """
    root, record, windows, sources, state = _ocular_record_context(
        project_root, record_plan, cache_root, record, index)
    pair_ids = list(selected_pairs)
    known = {p['id']:p for p in record_plan.get('pairs',[])}
    if not pair_ids or len(set(pair_ids)) != len(pair_ids) or set(pair_ids)-known.keys():
        raise ValueError('unique pair IDs from the retained parent record required')
    n, p = len(windows), len(pair_ids)
    output = {key:np.full((n,p,300),np.nan) for key in ('raw_proxy','clean_proxy','amplitude_proxy')}
    output.update(window_ids=np.asarray([w['id'] for w in windows]), pair_ids=np.asarray(pair_ids),
        time_s=np.arange(300)/10., complete=np.zeros((n,p),bool),
        amplitude_available=np.zeros((n,p),bool), frozen_pc=np.full((p,30),np.nan),
        frozen_eeg_factor=np.full(p,np.nan), channel_indices=np.full((p,6),-1,dtype=np.int64))
    metadata = dict(schema='tak_mismatch_eog_counterfactual_v1', status='unavailable', reason=None,
        record_id=record.join_key, source_identity=record_plan['source_identity'],
        planned_window_pairs=n*p, available_window_pairs=0, measured_signal_reads=0,
        fixed_parent_EEG_processing='native_EEG_power_without_project_cache_EOG_regression',
        intervention='existing_EOG_only_regression_on_native_EEG; same_reference_and_voltage_units',
        nuisance_fit_scope='full_native_record_offline_including_outside_scored_windows',
        coordinate_policy='unchanged_outer_training_PCA_sign_eeg_factor; no_PCA_or_Tak_refit',
        score_interval_s=list(SCORE_INTERVAL_S), amplitude_moments='score_bins_50_through_298_ddof0',
        interpretation='processing_sensitivity; not_independent_prediction_or_physiological_source_proof',
        pairs=[], windows=[], raw_reproduction=[])
    output['metadata'] = metadata
    if not state.get('auxiliary_channel_names'):
        metadata['reason'] = 'no_registered_native_EOG_auxiliary'
        return output
    retained = np.asarray(prepared['eeg_logpower'],float)
    if (retained.shape != (n,300,len(record_plan['eeg_channel_names']),5)
            or str(np.asarray(prepared['eeg_proxy_input']).item()) != EEG_PROXY_INPUT_SCHEMA
            or not np.array_equal(prepared['window_ids'],output['window_ids'])
            or not np.array_equal(prepared['time_s'],output['time_s'])):
        raise ValueError('retained parent EEG feature identity or 10 Hz clock mismatch')
    valid_pairs = []
    for j,pair_id in enumerate(pair_ids):
        pair = known[pair_id]
        coord = coordinate_models.get(pair_id,{})
        channels = pair.get('channel_sets',{}).get('nearest6')
        pc, factor = np.asarray(coord.get('pc',{}).get('nearest6',[]),float), coord.get('eeg_factor')
        row = dict(pair_id=pair_id,status='unavailable_frozen_coordinate')
        metadata['pairs'].append(row)
        if (channels is None or len(channels)!=6 or len(set(channels))!=6
                or any(c<0 or c>=retained.shape[2] for c in channels)
                or pc.shape!=(30,) or not np.isfinite(pc).all()
                or factor is None or not np.isfinite(factor) or factor<=0):
            continue
        output['channel_indices'][j] = channels
        output['frozen_pc'][j] = pc
        output['frozen_eeg_factor'][j] = factor
        row.update(status='available',channel_names=[record_plan['eeg_channel_names'][c] for c in channels])
        valid_pairs.append(j)
        for i,w in enumerate(windows):
            if w['status'] != 'available': continue
            features = retained[i][:,channels,:].reshape(300,30)
            if np.isfinite(features).all():
                output['raw_proxy'][i,j] = features@pc*factor
    if not valid_pairs or not np.isfinite(output['raw_proxy']).all(axis=2).any():
        metadata['reason']='no_complete_retained_raw_proxy'
        return output
    eeg=load_native_eeg_record(root,record)
    metadata['measured_signal_reads']=1
    _require_native_identity(eeg,record_plan,sources,state)
    names=[str(c).upper() for c in eeg.auxiliary_channel_names]
    if (eeg.auxiliary_values is None or any(names.count(c)!=1 for c in EOG_CHANNELS)
            or np.shape(eeg.auxiliary_values)!=(len(eeg.values),len(names))):
        metadata['reason']='native_HEOG_VEOG_pair_missing_or_ambiguous'
        return output
    rate=float(eeg.sample_rate_hz)
    selected=sorted(set(output['channel_indices'][valid_pairs].ravel().tolist()))
    voltage=np.asarray(eeg.values[:,selected],float)
    aux=np.asarray(eeg.auxiliary_values[:,[names.index(c) for c in EOG_CHANNELS]],float)
    metadata.update(native_unit=eeg.native_unit, unit_evidence=eeg.unit_evidence,
        native_sample_rate_hz=rate,native_samples=len(voltage),reference=eeg.reference,
        native_EOG_channels=list(EOG_CHANNELS),regression_channel_indices=selected)
    # Deterministic first/last usable windows check the original path without
    # recomputing every retained raw context. All selected channels are checked.
    check_indices=[i for i,w in enumerate(windows) if w['status']=='available'
                   and np.isfinite(output['raw_proxy'][i]).all(axis=1).any()]
    for i in sorted(set((check_indices[0],check_indices[-1]))):
        raw=_context_logpower(voltage,windows[i],rate)
        expected=retained[i][:,selected,:]
        if not np.allclose(raw,expected,rtol=0.,atol=1e-10,equal_nan=True):
            raise ValueError('native raw logpower does not reproduce retained parent features')
        finite=np.isfinite(expected)&np.isfinite(raw)
        feature_error=float(np.max(np.abs(raw[finite]-expected[finite]))) if finite.any() else None
        proxy_errors=[]
        for j in valid_pairs:
            local=[selected.index(c) for c in output['channel_indices'][j]]
            candidate=raw[:,local,:].reshape(300,30)@output['frozen_pc'][j]*output['frozen_eeg_factor'][j]
            target=output['raw_proxy'][i,j]
            if np.isfinite(target).all():
                if not np.allclose(candidate,target,rtol=0.,atol=1e-10):
                    raise ValueError('native raw proxy does not reproduce retained parent coordinate')
                proxy_errors.append(float(np.max(np.abs(candidate-target))))
        metadata['raw_reproduction'].append(dict(window_index=i,window_id=windows[i]['id'],
            maximum_feature_difference=feature_error,maximum_proxy_difference=max(proxy_errors,default=None),
            status='passed'))
    if not np.isfinite(aux).all():
        metadata['reason']='incomplete_native_record_EOG_support_no_regression_imputation'
        return output
    finite_channels=np.isfinite(voltage).all(axis=0)
    if not finite_channels.any():
        metadata['reason']='no_complete_native_record_EEG_channel'
        return output
    config=EEGArtifactCleaningConfig()
    mask,mask_state=detect_ocular_mask(aux,rate,config)
    clean=np.full_like(voltage,np.nan)
    clean[:,finite_channels],regression=_robust_eog_regression(
        voltage[:,finite_channels].copy(),aux,mask,rate,config)
    regression.update(channel_indices=[c for c,good in zip(selected,finite_channels) if good],
        variance_cap_reached=[bool(v>=config.max_removed_variance_fraction)
                              for v in regression['removed_variance_fraction']],
        raw_variance=np.var(voltage[:,finite_channels],axis=0).tolist(),
        clean_variance=np.var(clean[:,finite_channels],axis=0).tolist(),
        removed_variance_fraction_definition='variance_of_subtracted_regression_prediction_divided_by_raw_EEG_variance',
        coefficients_definition='existing_owner_coefficients_before_optional_prediction_variance_cap_scaling')
    keys=('eog_lag_s','eog_ridge','huber_delta','regression_iterations','max_regression_samples',
          'max_removed_variance_fraction','ocular_envelope_robust_z','ocular_velocity_robust_z','mask_dilation_s')
    metadata.update(regression=regression,ocular_weights=mask_state,
        regression_config={k:config.to_dict()[k] for k in keys},
        EOG_design='existing_0.5_to_15Hz_bandpass_median_MAD_lagged_EOG',
        EEG_extra_line_filter_bad_channel_or_muscle_processing='none')
    for i,w in enumerate(windows):
        if w['status']!='available': continue
        if not np.isfinite(output['raw_proxy'][i]).all(axis=1).any():
            metadata['windows'].extend(dict(window_id=w['id'],window_index=i,pair_id=pair_ids[j],
                status='unavailable',reason='incomplete_retained_raw_proxy') for j in valid_pairs)
            continue
        features=_context_logpower(clean,w,rate)
        for j in valid_pairs:
            row=dict(window_id=w['id'],window_index=i,pair_id=pair_ids[j],status='unavailable')
            metadata['windows'].append(row)
            if not np.isfinite(output['raw_proxy'][i,j]).all():
                row['reason']='incomplete_retained_raw_proxy'; continue
            local=[selected.index(c) for c in output['channel_indices'][j]]
            x=features[:,local,:].reshape(300,30)
            if not np.isfinite(x).all():
                row['reason']='incomplete_clean_native_context'; continue
            proxy=x@output['frozen_pc'][j]*output['frozen_eeg_factor'][j]
            output['clean_proxy'][i,j]=proxy
            output['complete'][i,j]=True
            matched,moments=amplitude_matched_proxy(output['raw_proxy'][i,j],proxy)
            output['amplitude_proxy'][i,j]=matched
            output['amplitude_available'][i,j]=np.isfinite(matched).all()
            row.update(status='available',reason=None,amplitude=moments)
    available=int(output['complete'].sum())
    metadata.update(status='available' if available else 'unavailable',
        reason=None if available else 'no_complete_clean_proxy',available_window_pairs=available)
    return output
