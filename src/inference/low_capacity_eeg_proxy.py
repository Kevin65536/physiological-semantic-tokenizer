"""Array-only low-capacity EEG readouts under the fixed Tak model.

This is a rest-tangent screening diagnostic, distinct from a nonlinear fit.
EEG observations retain the supplied temporal operator; no hidden Hb target is
accepted by the conditional predictor. Training membership, fold-local PCA and
scales, data access, and nonlinear confirmation remain the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq
from scipy.signal import periodogram

from .shared_driver_reconstruction import nonlinear_driver_forward


@dataclass(frozen=True)
class TakEEGConditionalMap:
    """Fixed EEG-to-Hb linear conditional map in processed coordinates.

    Reduced latent coordinates are [r[0:T],s0,delta_f0,delta_v0,delta_q0].
    They are rest tangents of the nonlinear log-initial coordinates. No p0
    coordinate or independent vascular innovation is introduced.
    """

    steps: int
    dt: float
    eeg_visible: np.ndarray
    observation_design: np.ndarray
    observation_offset: np.ndarray
    latent_map: np.ndarray
    latent_offset: np.ndarray
    regularizers: dict
    eeg_scale: np.ndarray
    data_weight: float
    initial_state_mode: str
    solve_method: str

    @property
    def hb_offset(self):
        return (self.observation_offset.ravel()
                + self.observation_design @ self.latent_offset).reshape(self.steps, 3)[:, 1:]

    def map_features(self, features):
        """Map processed EEG features [T,K] to Hb design [T,2,K].

        The offset is returned separately as ``hb_offset``. A linear readout
        ``features @ a`` yields ``hb_offset + einsum('tck,k->tc', D, a)``.
        No per-proxy centering or variance normalization is applied.
        """
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] != self.steps or not x.shape[1]:
            raise ValueError('features must be [T,K] on the frozen EEG grid')
        if not np.isfinite(x[self.eeg_visible]).all():
            raise ValueError('visible EEG features must be finite')
        latent = self.latent_map @ x[self.eeg_visible]
        return (self.observation_design @ latent).reshape(self.steps, 3, x.shape[1])[:, 1:]

    def predict(self, proxy):
        """Predict complete hidden Hb from EEG alone, retaining objective parts.

        A nonpositive linear initial estimate is reported without clipping.
        Linear validity and nonlinear physical-domain validity are separate.
        """
        eeg = np.asarray(proxy, dtype=float)
        if eeg.shape != (self.steps,) or not np.isfinite(eeg[self.eeg_visible]).all():
            raise ValueError('proxy must have finite visible EEG values [T]')
        latent = self.latent_map @ eeg[self.eeg_visible] + self.latent_offset
        prediction = (self.observation_offset.ravel()
                      + self.observation_design @ latent).reshape(self.steps, 3)
        initial = np.r_[0., np.ones(4)]
        if self.initial_state_mode == 'volume_tied':
            s, f, v, q = latent[self.steps:]
            initial += [s, f, v, v, q]
        residual = ((prediction[self.eeg_visible, 0] - eeg[self.eeg_visible])
                    / self.eeg_scale[self.eeg_visible])
        costs = dict(data=float(self.data_weight * (residual @ residual)))
        costs.update({name: float(np.sum((matrix @ latent)**2))
                      for name, matrix in self.regularizers.items()})
        return dict(prediction=prediction, driver=latent[:self.steps],
                    initial_state=initial, latent=latent, objective_components=costs,
                    objective=float(sum(costs.values())),
                    approximation='fixed_Tak_rest_tangent_EEG_only_conditional_screen',
                    initial_state_mode=self.initial_state_mode,
                    initial_free_parameters=4 if self.initial_state_mode == 'volume_tied' else 0)


def build_tak_eeg_conditional_map(parameters, steps, dt, *, mean_operator=None,
        sd=None, eeg_visible=None, penalty=.01, initial_penalty=100.,
        flow_prior_weight=1., flow_prior_log_sd=np.log(2.), data_weight=.4,
        substeps=4, viscoelastic_s=2., initial_state_mode='volume_tied',
        curvature_breaks=()):
    """Linearize the actual Tak RK4 map and solve the EEG-only prior problem.

    The objective matches the current nonlinear fitter's rest tangent: weighted
    processed EEG SSE, integrated driver curvature, four initial-state penalties
    (volume counted once), and integrated log-flow anchor. The same operator
    acts on the canonical EEG and Hb means *before* visibility selection.
    By default the final EEG bin is excluded, matching the 10 Hz contract.
    Hidden Hb never enters the right-hand side or a fitted coordinate scale.

    ``sd`` is frozen [3] or [T,3]. A caller using another dt must explicitly
    supply its matching data_weight; the adopted 10 Hz / .25 s value is .4.
    This is not the historical inlet-balance linear design.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 3:
        raise ValueError('steps must be an integer of at least three')
    if initial_state_mode not in ('volume_tied', 'rest'):
        raise ValueError('Tak initial_state_mode must be volume_tied or rest')
    controls = np.array([dt, penalty, initial_penalty, flow_prior_weight,
                         flow_prior_log_sd, data_weight], float)
    if (not np.isfinite(controls).all() or min(dt, flow_prior_log_sd, data_weight) <= 0
            or min(penalty, initial_penalty, flow_prior_weight) < 0):
        raise ValueError('invalid conditional-map objective controls')
    scale = np.ones((steps, 3)) if sd is None else np.asarray(sd, float)
    if scale.shape == (3,):
        scale = np.broadcast_to(scale, (steps, 3))
    if scale.shape != (steps, 3) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('sd must be frozen positive finite [3] or [T,3]')
    if eeg_visible is None:
        visible = np.ones(steps, bool)
        visible[-1] = False
    else:
        visible = np.asarray(eeg_visible)
    if visible.shape != (steps,) or visible.dtype != bool or not visible.any():
        raise ValueError('eeg_visible must be nonempty boolean [T]')
    breaks = list(curvature_breaks)
    if (any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) for b in breaks)
            or breaks != sorted(set(breaks)) or any(b < 3 or b > steps-3 for b in breaks)
            or any(b-a < 3 for a, b in zip(breaks, breaks[1:]))):
        raise ValueError('curvature_breaks requires ordered unique segment boundaries')
    forward = nonlinear_driver_forward(np.zeros(steps), np.r_[0., np.ones(4)],
        parameters, dt, substeps=substeps, derivative=True, return_flow_jacobian=True,
        hemodynamics='tak', viscoelastic_s=viscoelastic_s)
    # Tak's retained p0 derivative is exactly zero. Remove that column; the
    # remaining v0 column already differentiates both v and its derived p.
    columns = (np.r_[np.arange(steps+3), steps+4]
               if initial_state_mode == 'volume_tied' else np.arange(steps))
    design = forward['jacobian'][:, columns]
    offset = forward['canonical_prediction'].copy()
    if mean_operator is not None:
        operator = np.asarray(mean_operator, float)
        if operator.shape != (3*steps, 3*steps) or not np.isfinite(operator).all():
            raise ValueError('mean_operator must be finite [3T,3T]')
        design = operator @ design
        offset = (operator @ offset.ravel()).reshape(steps, 3)
    rows = np.arange(0, 3*steps, 3)[visible]
    weight = np.sqrt(data_weight)/scale[visible, 0]
    data_design = weight[:, None] * design[rows]
    count = len(columns)
    curvature = np.zeros((steps-2, count))
    curvature[:, :steps] = np.sqrt(penalty)*np.diff(np.eye(steps), n=2, axis=0)/dt**1.5
    for boundary in breaks:
        curvature[boundary-2:boundary] = 0.
    initial = np.zeros((4 if initial_state_mode == 'volume_tied' else 0, count))
    if initial_state_mode == 'volume_tied':
        initial[:, steps:] = np.sqrt(initial_penalty)*np.eye(4)
    flow = (np.sqrt(flow_prior_weight*dt)/flow_prior_log_sd
            * forward['flow_jacobian'][:, columns])
    regularizers = dict(driver=curvature, initial=initial, flow=flow)
    system = np.vstack((data_design, curvature, initial, flow))
    right = data_design.T * weight[None, :]
    try:
        factor = cho_factor(system.T @ system, lower=True, check_finite=False)
        latent_map = cho_solve(factor, right, check_finite=False)
        solve_method = 'cholesky_normal_equations'
    except np.linalg.LinAlgError:
        # With disabled priors the reference operator can leave a nullspace.
        # Retain the declared objective and its minimum-norm solution.
        rhs = np.zeros((len(system), int(visible.sum())))
        rhs[:len(data_design)] = np.diag(weight)
        latent_map = lstsq(system, rhs, lapack_driver='gelsd')[0]
        solve_method = 'minimum_norm_svd'
    latent_offset = -latent_map @ offset[visible, 0]
    return TakEEGConditionalMap(int(steps), float(dt), visible.copy(), design,
        offset, latent_map, latent_offset, regularizers, scale[:, 0].copy(),
        float(data_weight), initial_state_mode, solve_method)


def _regression_records(designs, targets, sd, offsets, masks):
    designs, targets = list(designs), list(targets)
    if not designs or len(designs) != len(targets):
        raise ValueError('paired nonempty training designs and targets are required')
    scales = np.asarray(sd, float)
    count = len(designs)
    offsets = [None]*count if offsets is None else list(offsets)
    masks = [None]*count if masks is None else list(masks)
    if len(offsets) != count or len(masks) != count:
        raise ValueError('one offset and mask per training record required')
    records = []
    width = None
    for i, (design, target, offset, mask) in enumerate(zip(designs, targets, offsets, masks)):
        x, y = np.asarray(design, float), np.asarray(target, float)
        if x.ndim != 3 or y.shape != x.shape[:2] or not x.shape[2]:
            raise ValueError('design [T,C,K] must match target [T,C]')
        if width is not None and width != x.shape[2]:
            raise ValueError('training records must have the same coefficient count')
        width = x.shape[2]
        off = np.zeros_like(y) if offset is None else np.asarray(offset, float)
        if off.shape != y.shape:
            raise ValueError('offset must match target [T,C]')
        visible = np.ones(y.shape, bool) if mask is None else np.asarray(mask)
        if visible.shape != y.shape or visible.dtype != bool or not visible.any():
            raise ValueError('training masks must be nonempty boolean [T,C]')
        scale = scales if scales.shape == (y.shape[1],) else scales[i] if scales.shape == (count, y.shape[1]) else None
        if scale is None or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('sd must be positive finite [C] or [record,C]')
        expanded = np.broadcast_to(scale, y.shape)[visible]
        xx, yy = x[visible]/expanded[:, None], (y[visible]-off[visible])/expanded
        if not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError('visible regression rows must be finite')
        records.append((xx, yy))
    return records


def fit_anchored_ridge(designs, targets, *, sd, anchor, ridge, offsets=None,
        masks=None, record_weights=None):
    """Fit a shared low-capacity readout without an intercept or output rescale.

    Use five ones for PC1 band gains, five zeros for spatial-mean readouts.
    Each supplied record contributes its mean standardized squared error;
    record_weights (e.g. inverse records per subject) may balance subjects.
    The penalty is ridge*trace(Gram)/K*||a-anchor||². It is dimensionless and
    remains in the caller's frozen EEG gauge. No target transform is fitted.
    """
    records = _regression_records(designs, targets, sd, offsets, masks)
    center = np.asarray(anchor, float)
    k = records[0][0].shape[1]
    if (center.shape != (k,) or not np.isfinite(center).all()
            or not np.isfinite(ridge) or ridge < 0):
        raise ValueError('finite coefficient anchor [K] and nonnegative ridge required')
    weights = np.ones(len(records)) if record_weights is None else np.asarray(record_weights, float)
    if weights.shape != (len(records),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('record_weights must be positive finite [record]')
    weights = weights/weights.sum()
    gram = sum(w*(x.T@x)/len(x) for w, (x, _) in zip(weights, records))
    cross = sum(w*(x.T@y)/len(x) for w, (x, y) in zip(weights, records))
    effective = float(ridge*np.trace(gram)/k)
    # Solve deviations from the anchor. This also retains the anchor in a
    # zero-information null direction when ridge=0.
    delta = lstsq(gram+effective*np.eye(k), cross-gram@center, lapack_driver='gelsd')[0]
    coefficients = center+delta
    data_cost = float(sum(w*np.mean((x@coefficients-y)**2)
                          for w, (x, y) in zip(weights, records)))
    prior_cost = float(effective*np.sum(delta**2))
    return dict(coefficients=coefficients, anchor=center.copy(), ridge=float(ridge),
                effective_ridge=effective, gram=gram, rank=int(np.linalg.matrix_rank(gram)),
                objective=data_cost+prior_cost, data_cost=data_cost, prior_cost=prior_cost,
                training_records=len(records), training_rows=sum(len(x) for x, _ in records),
                coefficient_count=k, intercept=False, output_renormalization=False,
                ridge_definition='ridge * trace(record_weighted_mean_Gram) / K')



def fixed_lag_fir_design(features, lags):
    """Causal fixed sample-lag features; invalid leading support remains NaN.

    Each column block is the complete input feature vector at t-lag. There
    is no wraparound, endpoint replication, or cross-record concatenation.
    Processed zero-phase EEG still makes this an offline information diagnostic.
    """
    x = np.asarray(features, float)
    if x.ndim == 1:
        x = x[:, None]
    lag_values = np.asarray(lags)
    if (x.ndim != 2 or not len(x) or not x.shape[1]
            or lag_values.ndim != 1 or not len(lag_values)
            or lag_values.dtype.kind not in 'iu' or np.any(lag_values < 0)
            or len(np.unique(lag_values)) != len(lag_values)
            or np.any(lag_values >= len(x))):
        raise ValueError('features [T,K] and unique nonnegative integer sample lags < T required')
    result = np.full((len(x), x.shape[1]*len(lag_values)), np.nan)
    for index, lag in enumerate(lag_values):
        result[lag:, index*x.shape[1]:(index+1)*x.shape[1]] = x[:len(x)-lag]
    valid = np.isfinite(result).all(axis=1)
    return result, valid



def residual_diagnostics(residual, dt, *, scale, mask=None, lags_s=(.5, 1., 2., 5.),
        regressors=None, spectral_bands_hz=((0., .05), (.05, .15), (.15, .5))):
    """Residual bias/RMS, species direction, lag correlations and spectra.

    Residual sign is target minus prediction. Masks preserve time gaps: lag
    pairs retain their original distance, and spectra average periodograms of
    complete contiguous segments without interpolation. Cross-component and
    regressor correlations use paired support; they are descriptive statistics,
    not independent-time-point significance tests or physiological attribution.
    """
    e, s = np.asarray(residual, float), np.asarray(scale, float)
    if (e.ndim != 2 or not len(e) or s.shape != (e.shape[1],)
            or not np.isfinite(s).all() or np.any(s <= 0) or not np.isfinite(dt) or dt <= 0):
        raise ValueError('residual [T,C], positive frozen scale [C] and dt required')
    visible = np.ones(e.shape, bool) if mask is None else np.asarray(mask)
    if visible.shape == (len(e),):
        visible = np.broadcast_to(visible[:, None], e.shape)
    if (visible.shape != e.shape or visible.dtype != bool or not visible.any(axis=0).all()
            or not np.isfinite(e[visible]).all()):
        raise ValueError('every residual component requires finite scored support')
    lags_seconds = np.asarray(lags_s, float)
    if lags_seconds.ndim != 1 or not np.isfinite(lags_seconds).all() or np.any(lags_seconds <= 0):
        raise ValueError('lags_s must be positive finite seconds')
    lags = np.rint(lags_seconds/dt).astype(int)
    if np.any(lags < 1):
        raise ValueError('lags_s must resolve to at least one sample')
    n_components = e.shape[1]
    bias = np.array([e[visible[:, c], c].mean() for c in range(n_components)])
    rms = np.array([np.sqrt(np.mean(e[visible[:, c], c]**2)) for c in range(n_components)])

    def correlation(x, y):
        if len(x) < 2:
            return np.nan
        x, y = x-x.mean(), y-y.mean()
        norm = np.linalg.norm(x)*np.linalg.norm(y)
        return float(x@y/norm) if norm > 0 else np.nan

    covariance, cross = np.full((n_components, n_components), np.nan), np.full((n_components, n_components), np.nan)
    for a in range(n_components):
        for b in range(n_components):
            paired = visible[:, a] & visible[:, b]
            x, y = e[paired, a], e[paired, b]
            if len(x):
                covariance[a, b] = np.mean((x-x.mean())*(y-y.mean()))
            cross[a, b] = correlation(x, y)
    acf, lag_counts = np.full((len(lags), n_components), np.nan), np.zeros((len(lags), n_components), int)
    for i, lag in enumerate(lags):
        if lag >= len(e):
            continue
        for c in range(n_components):
            paired = visible[:-lag, c] & visible[lag:, c]
            lag_counts[i, c] = int(paired.sum())
            acf[i, c] = correlation(e[:-lag, c][paired], e[lag:, c][paired])
    bands = np.asarray(spectral_bands_hz, float)
    if (bands.ndim != 2 or bands.shape[1] != 2 or not np.isfinite(bands).all()
            or np.any(bands[:, 0] < 0) or np.any(bands[:, 1] <= bands[:, 0])
            or np.any(bands[:, 1] > .5/dt)):
        raise ValueError('ordered spectral bands must lie within the Nyquist interval')
    power = np.full((n_components, len(bands)), np.nan)
    spectral_rows = np.zeros(n_components, int)
    # Use a shared FFT grid for all contiguous runs; do not invent samples in gaps.
    nfft = 2**int(np.ceil(np.log2(max(2, len(e)))))
    for c in range(n_components):
        edges = np.diff(np.r_[False, visible[:, c], False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        contributions, lengths = [], []
        for start, end in zip(starts, ends):
            if end-start < 3:
                continue
            frequency, density = periodogram(e[start:end, c], fs=1/dt,
                detrend='constant', scaling='density', nfft=nfft)
            df = frequency[1]-frequency[0]
            contributions.append([np.sum(density[(frequency >= lo) & (frequency < hi)])*df
                                  for lo, hi in bands])
            lengths.append(end-start)
        if lengths:
            power[c] = np.average(contributions, axis=0, weights=lengths)
            spectral_rows[c] = sum(lengths)
    result = dict(residual_sign='target_minus_prediction', count=visible.sum(axis=0),
        bias=bias, rms=rms, normalized_rms=rms/s, covariance=covariance,
        component_correlation=cross, lag_s=lags*dt, lag_samples=lags,
        lag_correlation=acf, lag_pair_count=lag_counts, spectral_bands_hz=bands,
        spectral_band_power=power, spectral_rows=spectral_rows,
        spectrum_rule='length_weighted_contiguous_segment_periodograms_demeaned',
        inference='descriptive_within_record_no_independent_timepoint_inference',
        cross_hb_phase='not_estimated')
    if n_components == 2:
        paired = visible.all(axis=1)
        oxygenated, deoxygenated = e[paired].T
        denom = oxygenated@oxygenated
        result['hbr_on_hbo_signed_projection'] = float(oxygenated@deoxygenated/denom) if denom else np.nan
        result['hbr_to_hbo_rms_ratio'] = float(rms[1]/rms[0]) if rms[0] else np.nan
    if regressors is not None:
        reg = np.asarray(regressors, float)
        if reg.ndim == 1:
            reg = reg[:, None]
        if reg.ndim != 2 or len(reg) != len(e):
            raise ValueError('regressors must preserve residual time support [T,R]')
        association = np.full((n_components, reg.shape[1]), np.nan)
        for c in range(n_components):
            for j in range(reg.shape[1]):
                paired = visible[:, c] & np.isfinite(reg[:, j])
                association[c, j] = correlation(e[paired, c], reg[paired, j])
        result['regressor_correlation'] = association
    return result
