"""Array-only mismatch diagnostics for the current four-state Tak model.

Coordinates are EEG, HbO, HbR; all supplied SDs are frozen training SDs.
These descriptive diagnostics neither read data nor assert physiological
identifiability. The observation operator acts on model predictions exactly
once, before masking and scoring. Historical inlet-balance diagnostics are
not used here.
"""
from __future__ import annotations

import numpy as np

from .shared_driver_reconstruction import (
    fit_nonlinear_shared_parameters,
    nonlinear_driver_forward,
)


def _positive_dt(dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    return float(dt)


def _scale(sd, size=3):
    value = np.asarray(sd, float)
    if value.shape != (size,) or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(f'{size} positive finite frozen training SDs required')
    return value


def _mask(values, valid):
    if valid is None:
        return np.isfinite(values)
    result = np.asarray(valid)
    if result.shape != values.shape or result.dtype != bool:
        raise ValueError('boolean mask must match observations')
    if not np.isfinite(values[result]).all():
        raise ValueError('visible observations must be finite')
    return result


def _operator(mean_operator, steps):
    if mean_operator is None:
        return None
    value = np.asarray(mean_operator, float)
    if value.shape != (3*steps, 3*steps) or not np.isfinite(value).all():
        raise ValueError('finite [3T,3T] mean_operator required')
    return value


def _processed(forward, operator):
    prediction = forward['canonical_prediction']
    return prediction.copy() if operator is None else (operator@prediction.ravel()).reshape(-1, 3)


def weak_balance_kernel(local_time, support_width_s):
    """Compact odd zero-integral kernel and its analytic time derivative.

    The complete support is ``[-support_width_s/2, support_width_s/2]``.
    Both kernel and derivative vanish at its endpoints. No derivative of
    measured Hb is taken.
    """
    if not np.isfinite(support_width_s) or support_width_s <= 0:
        raise ValueError('support width must be finite and positive')
    local_time = np.asarray(local_time, float)
    if not np.isfinite(local_time).all():
        raise ValueError('local time must be finite')
    half = float(support_width_s)/2
    u = local_time/half
    inside = np.abs(u) < 1
    phi = np.zeros_like(u)
    derivative = np.zeros_like(u)
    phi[inside] = u[inside]*(1-u[inside]**2)**4
    derivative[inside] = (1-u[inside]**2)**3*(1-9*u[inside]**2)/half
    return phi, derivative


def _weak_integral_weights(local_time, width):
    """Exact polynomial-kernel integral of piecewise-linear sampled Hb.

    Analytic integration, rather than a coarse quadrature of phi', preserves
    both zero moments even for constant reference shifts. The derivative
    weights follow integration by parts with zero endpoint kernel values.
    """
    half = width/2
    u = local_time/half
    coeff = np.array([0., 1., 0., -4., 0., 6., 0., -4., 0., 1.])
    primitive = np.polynomial.polynomial.polyint(coeff)
    first_moment = np.polynomial.polynomial.polyint(np.r_[0., coeff])
    integral = np.diff(np.polynomial.polynomial.polyval(u, primitive))*half
    moment = np.diff(np.polynomial.polynomial.polyval(u, first_moment))*half
    right = (moment-u[:-1]*integral)/np.diff(u)
    weights = np.zeros(len(u))
    weights[:-1] += integral-right
    weights[1:] += right
    derivative_weights = np.zeros(len(u))
    derivative_weights[:-1] += integral/np.diff(local_time)
    derivative_weights[1:] -= integral/np.diff(local_time)
    # Symmetry is exact mathematically; enforce it against polynomial roundoff.
    return (weights-weights[::-1])/2, (derivative_weights+derivative_weights[::-1])/2


def weak_balance_residual(hb, dt, parameters, *, valid=None,
                          support_widths_s=(10., 20., 40.), center_step_s=1.,
                          excitation_floor=0., viscoelastic_s=2.):
    """Small-signal HbT/HbR weak balance on complete supports only.

    ``hb`` is [T,2] in a common HbO/HbR observation scale, not separately
    standardized. Widths and center spacing must align with the sample grid;
    each width has an even number of intervals and its actual full endpoints
    are included. A missing sample invalidates the complete kernel support.
    ``excitation_floor`` compares with the mean denominator contribution
    ``2*mean(U**2+V**2)`` in the supplied Hb units and kernel convention;
    it is explicit and is never added to the denominator.
    """
    parameters.validate()
    dt = _positive_dt(dt)
    hb = np.asarray(hb, float)
    if hb.ndim != 2 or hb.shape[1] != 2 or len(hb) < 1:
        raise ValueError('hb must be [T,2] HbO/HbR')
    mask = _mask(hb, valid).all(axis=1)
    if (not np.isfinite([center_step_s, excitation_floor, viscoelastic_s]).all()
            or center_step_s <= 0 or excitation_floor < 0 or viscoelastic_s < 0):
        raise ValueError('invalid center spacing, excitation floor or viscoelasticity')
    stride = int(round(center_step_s/dt))
    if stride < 1 or not np.isclose(stride*dt, center_step_s, rtol=0, atol=1e-9):
        raise ValueError('center spacing must align with the sample grid')
    widths = tuple(float(v) for v in support_widths_s)
    if not widths or len(set(widths)) != len(widths):
        raise ValueError('distinct nonempty support widths required')
    alpha, e0 = parameters.fixed.alpha, parameters.fixed.E0
    tau = parameters.free.tau
    rho = parameters.fixed.Q0/parameters.fixed.P0
    eta = 1+(1-e0)*np.log1p(-e0)/e0
    a = eta*(tau+viscoelastic_s)-viscoelastic_s
    b = 1+(eta-1)/alpha
    result = {}
    for width in widths:
        if not np.isfinite(width) or width <= 0:
            raise ValueError('support widths must be finite and positive')
        intervals = int(round(width/dt))
        if intervals < 2 or intervals % 2 or not np.isclose(intervals*dt, width, rtol=0, atol=1e-9):
            raise ValueError('each full support must contain an even integer number of intervals')
        half = intervals//2
        local_time = np.arange(-half, half+1)*dt
        phi, dphi = weak_balance_kernel(local_time, width)
        wp, wd = _weak_integral_weights(local_time, width)
        first = ((half+stride-1)//stride)*stride
        centers = np.arange(first, len(hb)-half, stride, dtype=int)
        u = np.full(len(centers), np.nan)
        v = np.full(len(centers), np.nan)
        usable = np.zeros(len(centers), bool)
        for j, center in enumerate(centers):
            section = slice(center-half, center+half+1)
            if not mask[section].all():
                continue
            # Both analytic kernels have zero integral. Centering removes only
            # their null constant direction and avoids cancellation at large DC.
            pair = hb[section]-hb[center]
            total, deoxy = pair.sum(axis=1), pair[:, 1]
            u[j] = (wp-tau*wd)@deoxy
            v[j] = rho*(b*wp-a*wd)@total
            usable[j] = True
        residual = u-v
        denominator = float(2*np.sum(u[usable]**2+v[usable]**2))
        excitation = denominator/int(usable.sum()) if usable.any() else np.nan
        if not len(centers):
            status = 'unavailable_complete_support'
        elif not usable.any():
            status = 'unavailable_data_support'
        elif excitation <= excitation_floor:
            status = 'insufficient_dynamic_excitation'
        else:
            status = 'completed'
        score = float(np.sqrt(np.sum(residual[usable]**2)/denominator)) if status == 'completed' else np.nan
        result[width] = dict(status=status, D_bal=score, center_time_s=centers*dt,
            center_index=centers, U=u, V=v, w=residual, valid=usable,
            planned_centers=len(centers), valid_centers=int(usable.sum()),
            denominator=denominator, mean_denominator=excitation,
            excitation_floor=float(excitation_floor), support_width_s=width,
            kernel_time_s=local_time, phi=phi, phi_derivative=dphi)
    return dict(scales=result, coefficients=dict(eta=eta, A=a, B=b, rho=rho, tau=tau),
                interpretation='small_signal_descriptive_balance')


def _component_score(target, prediction, mask, sd):
    residual = np.full(target.shape, np.nan)
    count = int(mask.sum())
    if not count:
        status = 'unavailable_data_support'
    elif not np.isfinite(prediction[mask]).all():
        status = 'failed_prediction'
    else:
        status = 'completed'
        residual[mask] = target[mask]-prediction[mask]
    rmse = float(np.sqrt(np.mean(residual[mask]**2))) if status == 'completed' else np.nan
    return dict(status=status, count=count, rmse=rmse, normalized_rmse=rmse/sd,
                residual=residual, standardized_residual=residual/sd)


def score_tak_mismatch(observations, hb_prediction, joint_prediction, sd, *, visible=None):
    """D1, D_EH and D4 with shared Hb support and frozen coordinate SDs.

    Inputs are identically shaped [T,3] or [B,T,3]. Hb-only prediction also
    contains its EEG readout, including the same observation reference.
    Outputs aggregate only the explicitly supplied supports. Missing/failed
    results remain NaN rather than becoming zero mismatch.
    """
    y = np.asarray(observations, float)
    hp, jp = np.asarray(hb_prediction, float), np.asarray(joint_prediction, float)
    if y.ndim not in (2, 3) or y.shape[-1] != 3 or hp.shape != y.shape or jp.shape != y.shape:
        raise ValueError('observations and predictions must share [T,3] or [B,T,3] shape')
    scale = _scale(sd)
    mask = _mask(y, visible)
    ih = mask[..., 1]&mask[..., 2]
    ie = mask[..., 0]
    names = ('EEG', 'HbO', 'HbR')
    hs = {name: _component_score(y[..., c], hp[..., c], ie if c == 0 else ih, scale[c])
          for c, name in enumerate(names)}
    js = {name: _component_score(y[..., c], jp[..., c], ie if c == 0 else ih, scale[c])
          for c, name in enumerate(names)}
    d1 = float(np.sqrt(np.mean([hs[k]['normalized_rmse']**2 for k in names[1:]])))
    deh = hs['EEG']['normalized_rmse']
    d4 = float(np.sqrt(np.mean([js[k]['normalized_rmse']**2 for k in names])))
    signed = hs['EEG']['standardized_residual'][ie]
    ok = hs['EEG']['status'] == 'completed'
    statuses = [v['status'] for v in (*hs.values(), *js.values())]
    return dict(status='completed' if all(s == 'completed' for s in statuses) else 'incomplete',
        D1=d1, D_EH=deh, D4=d4, hb_only_components=hs, joint_components=js,
        D_J_E=js['EEG']['normalized_rmse'], D_J_O=js['HbO']['normalized_rmse'],
        D_J_R=js['HbR']['normalized_rmse'],
        D_EH_abs_p90=float(np.quantile(abs(signed), .9)) if ok else np.nan,
        D_EH_abs_p95=float(np.quantile(abs(signed), .95)) if ok else np.nan,
        hb_common_mask=ih, eeg_mask=ie, hb_time_count=int(ih.sum()),
        hb_observation_count=2*int(ih.sum()), eeg_count=int(ie.sum()))


def fit_tak_views(observations, parameters, dt, sd, *, mean_operator=None,
                  visible=None, parameter_names=(), parameter_bounds=(),
                  views=('hb_only', 'eeg_only', 'joint'), **fit_controls):
    """Fit declared modalities independently with target-independent starts.

    ``observations`` is [B,T,3]; physiological parameters, if requested,
    are shared across the entire supplied batch by the existing owner. Group
    definition is the caller's responsibility. All three views start from zero
    drive and resting four-state initial conditions; neither a hidden value
    nor a fit from another view can initialize a fit. Existing fitter control
    arguments are accepted except starts/SD/visibility/physiology dispatch.
    """
    y = np.asarray(observations, float)
    if y.ndim != 3 or y.shape[0] < 1 or y.shape[1] < 3 or y.shape[2] != 3:
        raise ValueError('observations must be [B,T,3]')
    _positive_dt(dt)
    scale = _scale(sd)
    mask = _mask(y, visible)
    blocked = {'starts', 'hemodynamics', 'sd', 'visible', 'parameter_names', 'parameter_bounds'}
    if blocked & fit_controls.keys():
        raise ValueError('starts and modality/physiology dispatch are controlled by fit_tak_views')
    if not views or len(set(views)) != len(views) or not set(views) <= {'hb_only', 'eeg_only', 'joint'}:
        raise ValueError('distinct supported fit views required')
    names = tuple(parameter_names)
    if names and set(names) != {'kappa', 'tau'}:
        raise ValueError('current_dof fits jointly share kappa and tau; fixed fits use no parameters')
    values = [getattr(parameters.free, name) for name in names]
    starts = [dict(parameter_values=values, driver=np.zeros(y.shape[:2]),
                   initial_state=np.tile([0., 1., 1., 1.], (len(y), 1)))]
    result = {}
    for view in views:
        selected = mask.copy()
        if view == 'hb_only':
            selected[..., 0] = False
        elif view == 'eeg_only':
            selected[..., 1:] = False
        absent = np.flatnonzero(~selected.any(axis=(1, 2)))
        if len(absent):
            result[view] = dict(status='unavailable_data_support', converged=False,
                                unavailable_batch_indices=absent)
            continue
        # Sanitize hidden coordinates before calling the numerical owner.
        result[view] = fit_nonlinear_shared_parameters(np.where(selected, y, np.nan),
            parameters, dt, parameter_names=names, parameter_bounds=parameter_bounds,
            sd=scale, mean_operator=mean_operator, visible=selected, starts=starts,
            hemodynamics='tak', **fit_controls)
    return result


def _hb_mask(valid, steps):
    if valid is None:
        return np.ones((steps, 2), bool)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.shape not in ((steps,), (steps, 2), (steps, 3)):
        raise ValueError('valid must be boolean [T], [T,2] or [T,3]')
    if valid.ndim == 1:
        return np.repeat(valid[:, None], 2, axis=1)
    pair = valid if valid.shape[1] == 2 else valid[:, 1:]
    return np.repeat(pair.all(axis=1)[:, None], 2, axis=1)


def _hb_scale(sd):
    scale = np.asarray(sd, float)
    return _scale(scale[1:] if scale.shape == (3,) else scale, 2)


def nonlinear_drive_gap(driver_e, driver_h, parameters, dt, sd, *,
                         mean_operator=None, initial_physical=None, valid=None,
                         substeps=4, viscoelastic_s=2.):
    """Nonlinear Hb replay gap at exactly the same parameters and initial state.

    Drives retain their supplied (typically regularizer-selected) DC gauge.
    The local observable diagnostic separately removes the EEG DC gauge.
    """
    re, rh = np.asarray(driver_e, float), np.asarray(driver_h, float)
    if re.ndim != 1 or rh.shape != re.shape or len(re) < 1:
        raise ValueError('matching nonempty [T] drivers required')
    operator = _operator(mean_operator, len(re))
    scale, mask = _hb_scale(sd), _hb_mask(valid, len(re))
    initial = np.array([0., 1., 1., 1.]) if initial_physical is None else np.asarray(initial_physical, float)
    predictions = []
    try:
        for driver in (re, rh):
            forward = nonlinear_driver_forward(driver, initial, parameters, dt,
                substeps=substeps, derivative=False, numerical_backend='numba',
                hemodynamics='tak', viscoelastic_s=viscoelastic_s)
            predictions.append(_processed(forward, operator))
    except (FloatingPointError, OverflowError) as exc:
        return dict(status='failed_physical_domain', D_drive_H=np.nan, error=str(exc),
                    hb_observation_count=int(mask.sum()))
    difference = predictions[0][:, 1:]-predictions[1][:, 1:]
    count = int(mask.sum())
    status = 'completed' if count else 'unavailable_data_support'
    return dict(status=status, D_drive_H=float(np.sqrt(np.mean((difference/scale)[mask]**2)))
                if count else np.nan, difference=difference,
                eeg_driver_prediction=predictions[0], hb_driver_prediction=predictions[1],
                valid=mask, hb_observation_count=count, initial_physical=initial.copy(),
                driver_dc_gauge='as_supplied_regularized_fit_gauge')


def observable_drive_gap(driver_e, driver_h, linearization_driver, initial_physical,
                         parameters, dt, sd, *, mean_operator=None, valid=None,
                         singular_value_threshold=None, relative_rank_tolerance=1e-10,
                         nuisance_parameters=(), remove_driver_dc=True,
                         substeps=4, viscoelastic_s=2.):
    """Local Hb-sensitive drive gap after projecting allowed nuisance changes.

    The discrete data Jacobian is evaluated at ``linearization_driver`` and
    the Hb-only initial state, transformed by the same observation operator
    and training SDs. Four initial coordinates are removed; optional kappa/tau
    columns are derivatives with respect to log parameters. No regularizer
    Jacobian is included. The constant driver direction is also removed by
    default because the current EEG reference cannot determine its DC gauge.
    The default returns D_obs=NaN and D_obs_allrank as an explicitly uncalibrated
    numerical-rank diagnostic. ``singular_value_threshold`` can be a scalar or
    one threshold per singular value, supplied by an external noise scenario.
    N_H is the number of visible scalar Hb observations, twice paired times.
    """
    re, rh, r0 = map(lambda v: np.asarray(v, float), (driver_e, driver_h, linearization_driver))
    if (r0.ndim != 1 or len(r0) < 1 or re.shape != r0.shape or rh.shape != r0.shape
            or not all(np.isfinite(v).all() for v in (re, rh, r0))):
        raise ValueError('three matching nonempty finite [T] drivers required')
    if not np.isfinite(relative_rank_tolerance) or relative_rank_tolerance <= 0:
        raise ValueError('positive relative_rank_tolerance required')
    nuisance_parameters = tuple(nuisance_parameters)
    if len(set(nuisance_parameters)) != len(nuisance_parameters) or not set(nuisance_parameters) <= {'kappa', 'tau'}:
        raise ValueError('distinct kappa/tau nuisance parameters required')
    scale, mask = _hb_scale(sd), _hb_mask(valid, len(r0))
    count = int(mask.sum())
    if not count:
        return dict(status='unavailable_data_support', D_obs=np.nan, D_obs_allrank=np.nan,
                    hb_observation_count=0,
                    singular_values=np.array([]), numerical_rank=0, retained_rank=0)
    operator = _operator(mean_operator, len(r0))
    try:
        forward = nonlinear_driver_forward(r0, initial_physical, parameters, dt,
            substeps=substeps, derivative=True, numerical_backend='numba',
            hemodynamics='tak', viscoelastic_s=viscoelastic_s)
    except (FloatingPointError, OverflowError) as exc:
        return dict(status='failed_physical_domain', D_obs=np.nan, D_obs_allrank=np.nan,
                    hb_observation_count=count, error=str(exc),
                    singular_values=np.array([]), numerical_rank=0, retained_rank=0)
    jac = forward['jacobian']
    if operator is not None:
        jac = operator@jac
    shaped = jac.reshape(len(r0), 3, -1)[:, 1:, :]
    weighted = (shaped/scale[None, :, None])[mask]
    jr = weighted[:, :len(r0)]
    nuisance = weighted[:, len(r0)+np.array([0, 1, 2, 4])]
    if not isinstance(remove_driver_dc, (bool, np.bool_)):
        raise ValueError('remove_driver_dc must be boolean')
    if remove_driver_dc:
        nuisance = np.column_stack((nuisance, jr@np.ones(len(r0))))
    for name in nuisance_parameters:
        tangent = forward[name+'_jacobian']*getattr(parameters.free, name)
        if operator is not None:
            tangent = (operator@tangent.ravel()).reshape(-1, 3)
        nuisance = np.column_stack((nuisance, (tangent[:, 1:]/scale)[mask]))
    un, sn, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = int(np.sum(sn > relative_rank_tolerance*sn[0])) if len(sn) and sn[0] > 0 else 0
    un = un[:, :nuisance_rank]
    projected = jr-un@(un.T@jr)
    u, singular, vh = np.linalg.svd(projected, full_matrices=False)
    # Scale against the unprojected design so cancellation of a fully nuisance
    # span cannot promote roundoff into observable modes.
    rank_cutoff = relative_rank_tolerance*np.linalg.norm(jr, ord='fro')
    numerical = singular > rank_cutoff
    threshold = np.asarray(0. if singular_value_threshold is None else singular_value_threshold, float)
    if (threshold.ndim > 1 or threshold.shape not in ((), singular.shape)
            or not np.isfinite(threshold).all() or np.any(threshold < 0)):
        raise ValueError('finite nonnegative scalar or per-mode singular threshold required')
    retained = numerical & (singular > threshold)
    coordinates = vh@(re-rh)
    all_rank_gap = (float(np.linalg.norm(singular[numerical]*coordinates[numerical])/np.sqrt(count))
                    if numerical.any() else np.nan)
    gap = (float(np.linalg.norm(singular[retained]*coordinates[retained])/np.sqrt(count))
           if retained.any() and singular_value_threshold is not None else np.nan)
    status = ('uncalibrated_local_diagnostic' if singular_value_threshold is None
              else 'thresholded_local_diagnostic') if retained.any() else 'insufficient_observable_rank'
    return dict(status=status, D_obs=gap, D_obs_allrank=all_rank_gap,
        singular_values=singular, numerical_rank=int(numerical.sum()),
        numerical_rank_cutoff=float(rank_cutoff),
        retained_rank=int(retained.sum()), retained_modes=retained,
        singular_value_threshold=threshold, nuisance_rank=nuisance_rank,
        nuisance_singular_values=sn, nuisance_parameters=nuisance_parameters,
        remove_driver_dc=bool(remove_driver_dc),
        hb_observation_count=count, hb_common_mask=mask,
        left_singular_vectors=u, right_singular_vectors=vh,
        projected_driver_jacobian=projected, driver_mode_coordinates=coordinates,
        interpretation='local_noise_sensitivity_not_measured_identifiability')


def _validate_noise_settings(driver_train_sd, driver_steps, noise_sd, ar1,
                             cross_correlation, draws, quantile):
    if (not np.isfinite([driver_train_sd, noise_sd, ar1, cross_correlation, quantile]).all()
            or driver_train_sd <= 0 or noise_sd <= 0 or abs(ar1) >= 1
            or abs(cross_correlation) >= 1 or not 0 < quantile < 1):
        raise ValueError('invalid driver/noise scale, correlation or quantile')
    for value, minimum in ((driver_steps, 1), (draws, 2)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError('positive integer driver_steps and at least two draws required')


def _noise_mode_quantile(observable, driver_steps, ar1, cross_correlation,
                         draws, seed, quantile, standard_normals=None):
    """Project one unit-stationary-SD noise covariance group only once."""
    mask = np.asarray(observable['hb_common_mask'])
    if mask.shape != (driver_steps, 2):
        raise ValueError('driver_steps must match observable time support')
    u = np.asarray(observable['left_singular_vectors'], float)
    singular = np.asarray(observable['singular_values'], float)
    if u.shape != (int(mask.sum()), len(singular)):
        raise ValueError('observable singular vectors do not match Hb support')
    if standard_normals is None:
        standard_normals = np.random.default_rng(seed).normal(size=(draws, driver_steps, 2))
    noise = standard_normals.copy()
    noise[..., 1] = cross_correlation*noise[..., 0]+np.sqrt(1-cross_correlation**2)*noise[..., 1]
    for t in range(1, driver_steps):
        noise[:, t] = ar1*noise[:, t-1]+np.sqrt(1-ar1**2)*noise[:, t]
    projected_noise = noise[:, mask]@u
    return np.quantile(abs(projected_noise), quantile, axis=0)


def _calibrated_mode_result(observable, driver_train_sd, driver_steps, threshold,
                            noise_sd, ar1, cross_correlation, draws, seed, quantile):
    mask = np.asarray(observable['hb_common_mask'])
    singular = np.asarray(observable['singular_values'])
    amplitude = driver_train_sd*np.sqrt(driver_steps)
    numerical_rank = int(observable['numerical_rank'])
    retained = (amplitude*singular > threshold)&(np.arange(len(singular)) < numerical_rank)
    coordinates = np.asarray(observable['driver_mode_coordinates'])
    gap = float(np.linalg.norm((singular*coordinates)[retained])/np.sqrt(mask.sum())) if retained.any() else np.nan
    return dict(status='synthetic_noise_sensitivity' if retained.any() else 'insufficient_observable_rank',
        D_obs=gap, retained_modes=retained, retained_rank=int(retained.sum()),
        projected_noise_threshold=threshold, singular_value_threshold=threshold/amplitude,
        driver_mode_perturbation_norm=amplitude, driver_train_sd=float(driver_train_sd),
        noise_sd=float(noise_sd), ar1=float(ar1), cross_correlation=float(cross_correlation),
        draws=int(draws), seed=seed, quantile=float(quantile),
        noise_coordinate='standardized_processed_Hb',
        interpretation='synthetic_noise_sensitivity_not_measured_identifiability')


def calibrate_observable_modes(observable, *, driver_train_sd, driver_steps,
                               noise_sd=.1, ar1=0., cross_correlation=.3,
                               draws=256, seed=0, quantile=.95):
    """Frozen synthetic standardized-Hb noise sensitivity for local modes.

    Noise is generated on the full time grid, then the same Hb mask is used.
    Each chromophore has the declared stationary SD, AR(1) coefficient and
    simultaneous cross-correlation. Thresholds are quantiles of
    abs(u_k.T @ standardized_noise). A mode survives if a_r*sqrt(T)*s_k is
    larger; no measured noise estimate or identifiability claim is implied.
    """
    _validate_noise_settings(driver_train_sd, driver_steps, noise_sd, ar1,
                             cross_correlation, draws, quantile)
    threshold = noise_sd*_noise_mode_quantile(observable, driver_steps, ar1,
        cross_correlation, draws, seed, quantile)
    return _calibrated_mode_result(observable, driver_train_sd, driver_steps,
        threshold, noise_sd, ar1, cross_correlation, draws, seed, quantile)


_DEFAULT_NOISE_SCENARIOS = (
    ('primary_ar1_sd0p1', .1, .9, .3),
    ('ar1_sd0p05', .05, .9, .3),
    ('ar1_sd0p2', .2, .9, .3),
    ('iid_sd0p1', .1, 0., 0.),
)


def observable_drive_scenarios(driver_e, driver_h, linearization_driver,
                               initial_physical, parameters, dt, sd, *,
                               driver_train_sd, noise_scenarios=None,
                               draws=256, seed=0, quantile=.95,
                               **observable_controls):
    """One data-Jacobian SVD with explicit synthetic-noise sensitivity scenarios.

    Positional arguments and ``observable_controls`` match
    :func:`observable_drive_gap`. ``noise_scenarios`` is a nonempty sequence
    of dictionaries with id, noise_sd, ar1 and cross_correlation; its first
    entry is the primary scenario. Defaults are standardized processed-Hb
    SD .1 / AR(1) .9 / Hb cross-correlation .3, SD .05 and .2 sensitivities,
    and iid SD .1 / cross-correlation 0. These are declared assumptions,
    never estimates from measured fit residuals.

    The same frozen standard-normal draws underlie all scenarios. Only two
    noise projections are needed for the default four scenarios: scalar SD
    changes reuse and scale the same per-mode quantiles. Returned scenario
    curves are standardized Hb output differences retained by that scenario,
    on the original paired time support, with NaN outside that support.
    Their RMS over the N_H scalar observations equals scenario D_obs.
    """
    if 'singular_value_threshold' in observable_controls:
        raise ValueError('scenario thresholds are determined by declared synthetic noise')
    driver_steps = len(np.asarray(linearization_driver))
    scenarios = ([dict(id=name, noise_sd=level, ar1=ar, cross_correlation=corr)
                  for name, level, ar, corr in _DEFAULT_NOISE_SCENARIOS]
                 if noise_scenarios is None else [dict(value) for value in noise_scenarios])
    if not scenarios:
        raise ValueError('at least one named noise scenario required')
    ids = [value.get('id') for value in scenarios]
    if any(not isinstance(name, str) or not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError('distinct nonempty noise scenario ids required')
    for scenario in scenarios:
        if not {'noise_sd', 'ar1', 'cross_correlation'} <= scenario.keys():
            raise ValueError('each noise scenario requires noise_sd, ar1 and cross_correlation')
        _validate_noise_settings(driver_train_sd, driver_steps, scenario['noise_sd'],
            scenario['ar1'], scenario['cross_correlation'], draws, quantile)
    observable = observable_drive_gap(driver_e, driver_h, linearization_driver,
        initial_physical, parameters, dt, sd, **observable_controls)
    rank = observable['numerical_rank']
    mode_curves = {}
    thresholds = {}
    results = {}
    if rank:
        standard_normals = np.random.default_rng(seed).normal(size=(draws, driver_steps, 2))
        mask = observable['hb_common_mask']
        singular = observable['singular_values']
        coordinates = observable['driver_mode_coordinates']
        for scenario in scenarios:
            covariance = (float(scenario['ar1']), float(scenario['cross_correlation']))
            if covariance not in thresholds:
                thresholds[covariance] = _noise_mode_quantile(observable, driver_steps,
                    *covariance, draws, seed, quantile, standard_normals=standard_normals)
            threshold = float(scenario['noise_sd'])*thresholds[covariance]
            result = _calibrated_mode_result(observable, driver_train_sd, driver_steps,
                threshold, scenario['noise_sd'], *covariance, draws, seed, quantile)
            curve = np.full((driver_steps, 2), np.nan)
            if result['retained_rank']:
                keep = result['retained_modes']
                curve[mask] = observable['left_singular_vectors'][:, keep]@(singular*coordinates)[keep]
            result['standardized_hb_difference'] = curve
            result['id'] = scenario['id']
            results[scenario['id']] = result
        mode_curves = dict(singular_values=singular,
            mode_signal_at_training_driver_rms=driver_train_sd*np.sqrt(driver_steps)*singular,
            actual_mode_output_difference=singular*coordinates)
    else:
        # Missing support, failed physics and rank zero must remain unavailable;
        # no scenario can turn them into a numerical zero score.
        for scenario in scenarios:
            results[scenario['id']] = dict(scenario, status=observable['status'], D_obs=np.nan,
                retained_rank=0, retained_modes=np.zeros(len(observable['singular_values']), bool),
                standardized_hb_difference=np.full((driver_steps, 2), np.nan),
                driver_train_sd=float(driver_train_sd), draws=int(draws), seed=seed,
                quantile=float(quantile), noise_coordinate='standardized_processed_Hb',
                interpretation='synthetic_noise_sensitivity_not_measured_identifiability')
    primary = results[ids[0]]
    return dict(status=primary['status'], D_obs=primary['D_obs'],
        D_obs_allrank=observable['D_obs_allrank'], numerical_rank=rank,
        retained_rank=primary['retained_rank'], primary_noise_scenario=ids[0],
        scenarios=results, observable=observable, mode_curves=mode_curves,
        noise_projection_groups=len(thresholds),
        noise_seed_policy='same_frozen_standard_normal_draws_across_scenarios',
        interpretation='synthetic_noise_sensitivity_not_measured_identifiability')
