"""Information-destruction controls for frozen local spatial reconstructions.

This owner contains no dataset discovery, raw reader, or protected-data path.
Inputs use the parent run's train-frozen observation coordinate. A shift acts
on already referenced features, so it is not a physiological delay experiment.
"""
from __future__ import annotations

from itertools import product
import numpy as np

from .observation_baselines import ridge_predict, transform_spatial_coordinates
from .shared_driver_spatial import (
    fit_spatial_shared_driver, forward_spatial_shared_driver, temporal_mode_basis,
)
from .t3a_balloon_robust_ssm import BalloonFixedParameters, BalloonFreeParameters, BalloonParameters
from .source_observability import weighted_jacobian


def physical_parameters(config):
    fixed = dict(config['fixed'])
    tau, kappa = fixed.pop('tau'), fixed.pop('kappa')
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=tau, kappa=kappa))


def scaled_record(record, coordinate, *, hide_target=True, eeg_source_steps=None):
    """Mask evaluation target before even validating the frozen transform.

    ``hide_target=False`` is reserved for explicit train-background construction.
    The scoring path separately reads the target and never passes it here.
    """
    h = np.array(record['hb'], dtype=float, copy=True)
    if hide_target:
        h[:, 0] = np.nan
    e = np.asarray(record['eeg'], float)
    if eeg_source_steps is not None:
        keep = np.zeros(len(e), bool)
        indices = np.asarray(eeg_source_steps, int)
        if np.any(indices < 0) or np.any(indices >= len(e)):
            raise ValueError('EEG source support outside real feature rows')
        keep[indices] = True
        e = np.where(keep.reshape((len(e),)+(1,)*(e.ndim-1)), e, np.nan)
    e, h = transform_spatial_coordinates(e, h, coordinate)
    return dict(eeg=e, hb=h, eeg_operator=np.asarray(record['eeg_operator']),
                hb_operator=np.asarray(record['hb_operator']))



def validate_wrong_partners(rows, partners):
    """Validate bounded parent pairing metadata before any donor array read."""
    lookup = {row['id']: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('duplicate parent identity')
    patch_keys = ('eeg_channel_names', 'hb_pair_names', 'coordinate_system',
                  'eeg_coordinates', 'hb_coordinates')
    for row in rows:
        partner = partners.get(row['id'])
        if partner is None:
            continue
        if partner not in lookup or partner == row['id']:
            raise ValueError('wrong EEG partner missing or identical to target')
        other = lookup[partner]
        if any(row.get(key) != other.get(key) for key in ('dataset', 'subject', 'task', 'condition', 'split', 'window_seconds')):
            raise ValueError('wrong EEG partner violates subject/task/split match')
        if any(row['patch'].get(key) != other['patch'].get(key) for key in patch_keys):
            raise ValueError('wrong EEG partner violates verified montage match')
        if row['record'] == other['record'] and any(abs(row[key]-other[key]) < row['window_seconds'] for key in ('eeg_start_s', 'hb_start_s')):
            raise ValueError('wrong EEG partner must be nonoverlapping')
    return True


def perturb_inputs(record, condition, *, wrong=None, support=(32, 88), shift_steps=None):
    """Mask target before assembly; shifts index only existing source rows.

    EEG support is identical for all positive-weight arms. All donor Hb rows
    remain visible. Target pair 0 is absent at every time, including reference.
    A wrong donor's Hb is never indexed, transformed, or passed to a fitter.
    """
    e = np.asarray(record['eeg'], float)
    h = np.asarray(record['hb'], float)
    t, f = e.shape
    left, right = support
    if h.shape != (t, 4, 2) or not 0 <= left < right <= t:
        raise ValueError('expected common [T,60] EEG and [T,4,2] Hb support')
    visible = np.ones((t, f+8), bool)
    visible[:, f:f+2] = False
    visible[:, :f] = False
    x = np.full_like(e, np.nan)
    if condition != 'Hb_only':
        source = np.asarray(wrong['eeg'] if condition == 'wrong_EEG' and wrong is not None else e, float)
        if condition == 'wrong_EEG' and wrong is None:
            raise ValueError('matched wrong-EEG donor unavailable')
        lag = (shift_steps or {}).get(condition, 0)
        if condition not in ('paired', 'wrong_EEG') and condition not in (shift_steps or {}):
            raise ValueError('unregistered feature perturbation')
        indices = np.arange(left, right)+lag
        if source.shape != e.shape or np.any(indices < 0) or np.any(indices >= t):
            raise ValueError('noncircular EEG shift must remain within real support')
        x[left:right] = source[indices]
        visible[left:right, :f] = True
    hidden_h = np.where(visible[:, f:].reshape(h.shape), h, np.nan)
    if not np.isfinite(x[visible[:, :f]]).all() or not np.isfinite(hidden_h[:, 1:]).all():
        raise ValueError('visible EEG and donor Hb must be finite')
    return x, hidden_h, visible


def forward_kwargs(parent_config, record, calibration):
    return dict(eeg_loadings=np.asarray(calibration['readout']['eeg_loadings']),
        hb_loadings=np.asarray(calibration['design']['hb_loadings']),
        temporal_basis=temporal_mode_basis(len(record['eeg']), parent_config['tensor']['temporal_modes']),
        eeg_operator=record['eeg_operator'], hb_operator=record['hb_operator'],
        numerical_backend=parent_config['solver']['numerical_backend'])


def fit_condition(record, calibration, coordinate, parent_config, condition, *, wrong=None,
                  support=(32, 88), shift_steps=None):
    e, h, visible = perturb_inputs(record, condition, wrong=wrong, support=support, shift_steps=shift_steps)
    controls = parent_config['solver']
    kwargs = forward_kwargs(parent_config, record, calibration)
    fit = fit_spatial_shared_driver(e, h, physical_parameters(parent_config),
        parent_config['tensor']['dt_s'], **kwargs, sd=np.asarray(coordinate['sd']),
        visible=visible, eeg_weight=0. if condition == 'Hb_only' else 1.,
        driver_curvature_weight=controls['driver_curvature_weight'],
        initial_weight=controls['initial_weight'], flow_prior_weight=controls['flow_prior_weight'],
        flow_log_sd=controls['flow_log_sd'], max_nfev=controls['max_nfev'],
        gradient_tolerance=controls['gradient_tolerance'], substeps=controls['substeps'])
    fit['visible'] = visible
    if fit['success']:
        replay = forward_spatial_shared_driver(fit['coefficients'], fit['initial_coordinates'],
            physical_parameters(parent_config), parent_config['tensor']['dt_s'], **kwargs,
            derivative=False, substeps=controls['validation_substeps'])
        error = float(np.max(np.abs(replay['prediction']-fit['prediction'])/np.asarray(coordinate['sd'])))
        fit['integration_max_training_sd'] = error
        if error > controls['integration_tolerance_training_sd']:
            fit.update(success=False, status='failed_integration')
    return fit


def predict_ridge(record, calibration, condition, *, wrong=None, support=(32, 88), shift_steps=None):
    """Frozen raw controls with precisely the same available covariates.

    Joint training ridge applies where EEG is visible; elsewhere the frozen
    donor-only spatial ridge applies. No target value enters this operation.
    """
    e, h, visible = perturb_inputs(record, condition, wrong=wrong, support=support, shift_steps=shift_steps)
    donors = h[:, 1:].reshape(len(h), -1)
    models = calibration['baselines']
    def model(name):
        return {k: np.asarray(v) if isinstance(v, list) else v for k, v in models[name].items()}
    prediction = ridge_predict(model('spatial_ridge__Hb_only'), donors)
    take = visible[:, :e.shape[1]].all(axis=1)
    if take.any():
        prediction[take] = ridge_predict(model('joint_ridge__paired'), np.column_stack((e[take], donors[take])))
    return prediction


def score_target(prediction, target, sd, support=(20, 120)):
    p, y, sd = np.asarray(prediction), np.asarray(target), np.asarray(sd)
    left, right = support
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 2 or sd.shape != (2,):
        raise ValueError('target score requires [T,2] and frozen two-component SD')
    if not (0 <= left < right <= len(y)) or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError('invalid fixed scoring support or scale')
    error = (p[left:right]-y[left:right])/sd
    if not np.isfinite(error).all():
        raise ValueError('nonfinite target endpoint')
    component = np.mean(error**2, axis=0)
    return dict(nmse=float(np.mean(component)), nrmse=float(np.sqrt(np.mean(component))),
        component_nmse=component, scored_values=2*(right-left),
        denominator='parent_frozen_training_SD_per_Hb_component')


def exact_sign_flip(values):
    """Exact two-sided paired subject sign test of the mean signed difference."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or len(values) > 20 or not np.isfinite(values).all():
        raise ValueError('1..20 finite independent-unit differences required')
    observed = abs(float(values.mean()))
    means = np.asarray([abs(float(np.mean(values*np.asarray(signs))))
                        for signs in product((-1., 1.), repeat=len(values))])
    return float(np.mean(means >= observed-1e-14))


def paired_statistics(left, right, *, repeats=2000, seed=0, metric='nrmse'):
    """Positive means candidate(left) has lower subject NRMSE than reference."""
    units = sorted(set(left) & set(right))
    if not units:
        return dict(status='unavailable', independent_units=0)
    delta = np.asarray([right[x]-left[x] for x in units])
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (repeats, len(delta)))].mean(axis=1)
    return dict(status='computed', independent_units=len(units), units=units,
        metric=metric, **{'candidate_'+metric: {x: left[x] for x in units},
                        'reference_'+metric: {x: right[x] for x in units}},
        delta_reference_minus_candidate=float(delta.mean()),
        bootstrap_percentile95=np.quantile(means, [.025, .975]),
        exact_two_sided_sign_flip_p=exact_sign_flip(delta),
        interpretation='descriptive_frozen_development_subject_pairs_not_fresh_confirmation')


def local_added_constraint(fit, record, calibration, coordinate, parent_config, *, support=(32, 88)):
    """Data-only Jacobian at Hb-only solution, with frozen unit coordinates.

    Report EEG sensitivity along the bottom quarter right-singular directions
    of donor Hb. These are local parameter directions, not identified sources.
    """
    kwargs = forward_kwargs(parent_config, record, calibration)
    forward = forward_spatial_shared_driver(fit['coefficients'], fit['initial_coordinates'],
        physical_parameters(parent_config), parent_config['tensor']['dt_s'], **kwargs,
        derivative=True, substeps=parent_config['solver']['substeps'])
    t, f = np.asarray(record['eeg']).shape
    jac = forward['jacobian'].reshape(t, f+8, -1)
    # Match the fitter's modality mean-SSE weights. No regularizer rows enter.
    hb = weighted_jacobian(jac[:, f+2:], np.asarray(coordinate['sd'])[f+2:],
        feature_weights=np.full(6, 1/np.sqrt(6.)))
    left, right = support
    eeg = weighted_jacobian(jac[left:right, :f], np.asarray(coordinate['sd'])[:f],
        feature_weights=np.full(f, np.sqrt(t/((right-left)*f))))
    _, singular, vt = np.linalg.svd(hb, full_matrices=False)
    count = max(1, int(np.ceil(len(singular)/4)))
    weak = vt[-count:].T
    eeg_energy = np.sum((eeg@weak)**2, axis=0)
    hb_energy = np.sum((hb@weak)**2, axis=0)
    return dict(unknowns=len(singular), weak_directions=count,
        hb_singular_values=singular, weak_hb_energy=hb_energy,
        added_EEG_energy_in_weak_directions=eeg_energy,
        weak_direction_EEG_energy_fraction=float(eeg_energy.sum()/max(np.sum(eeg**2), 1e-30)),
        coordinate='fixed_parent_coefficients_and_log_initial_coordinates_no_posthoc_rescale',
        regularizer_rows_included=False, actual_EEG_values_used=False,
        weak_direction_definition='bottom_ceil_P_over4_Hb_right_singular_directions_not_exact_nullspace',
        unchanged_by_wrong_or_shifted_EEG_values_at_same_linearization=True,
        interpretation='forward_readout_constraint_potential_only_not_observed_independent_EEG_information_or_neural_recovery')
