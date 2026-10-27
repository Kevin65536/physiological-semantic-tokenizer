"""Synthetic-only Tak EEG screening, train-only readouts and residual diagnostics."""
import numpy as np
import pytest
from scipy.linalg import lstsq

from src.inference.low_capacity_eeg_proxy import (
    build_tak_eeg_conditional_map, fit_anchored_ridge,
    fixed_lag_fir_design, residual_diagnostics,
)
from src.inference.shared_driver_reconstruction import (
    _nonlinear_trial_objective, nonlinear_driver_forward,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def reference_operator(n):
    reference = np.eye(n)
    reference[:, :5] -= 1/5
    # Keep EEG's reference and a fixed Hb observation mixing in the same
    # operator; neither is replaced by an arbitrary raw-driver convolution.
    hb_map = np.array([[1., 0., 0.], [0., 1.2, -.1], [0., -.2, .8]])
    return np.kron(reference, hb_map)


@pytest.mark.parametrize('mode', ['volume_tied', 'rest'])
def test_tak_conditional_map_matches_full_nonlinear_residual_tangent(mode):
    n, dt = 24, .1
    p = BalloonParameters()
    operator = reference_operator(n)
    sd = np.array([.025, .03, .02])
    mapper = build_tak_eeg_conditional_map(p, n, dt, mean_operator=operator,
        sd=sd, initial_state_mode=mode)
    # Build the actual nonlinear objective at rest, with EEG-only visibility.
    mask = np.zeros((n, 3), bool)
    mask[:-1, 0] = True
    full = np.eye(n+5)
    if mode == 'volume_tied':
        embedding = full[:, np.r_[np.arange(n+3), n+4]]
        embedding[n+3, n+2] = 1.
    else:
        embedding = full[:, :n]
    curvature = np.zeros((n-2, n+5))
    curvature[:, :n] = np.sqrt(.01)*np.diff(np.eye(n), n=2, axis=0)/dt**1.5
    trial = _nonlinear_trial_objective(np.zeros(embedding.shape[1]), p, dt,
        operator, np.zeros((n, 3)), mask, np.broadcast_to(sd, (n, 3)),
        curvature, 100., 4, True, parameter_name=[], initial_embedding=embedding,
        numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.,
        initial_coordinate_weights=[1, 1, 1, 0, 1], flow_prior_weight=1.,
        flow_prior_log_sd=np.log(2.), data_weight=.4)
    proxy = .02*np.sin(np.arange(n)*.37)
    rhs = np.zeros(len(trial['residual']))
    rhs[:n-1] = np.sqrt(.4)*proxy[:-1]/sd[0]
    expected = lstsq(trial['jacobian'], rhs)[0]
    actual = mapper.predict(proxy)
    np.testing.assert_allclose(actual['latent'], expected, atol=3e-12, rtol=1e-9)
    np.testing.assert_allclose(actual['objective'], np.sum((trial['jacobian']@expected-rhs)**2), atol=2e-12)
    np.testing.assert_equal(actual['initial_state'][2], actual['initial_state'][3])
    assert actual['initial_free_parameters'] == (4 if mode == 'volume_tied' else 0)


def test_processed_eeg_is_not_used_as_raw_driver_and_hidden_last_bin_cannot_enter():
    n = 40
    operator = reference_operator(n)
    mapper = build_tak_eeg_conditional_map(BalloonParameters(), n, .1,
        mean_operator=operator, sd=[.025, .03, .02])
    features = np.random.default_rng(4).normal(size=(n, 5))*.02
    original = features.sum(axis=1)
    fit = mapper.predict(original)
    # Reference removes the EEG constant; flow/initial priors choose its level.
    shifted = mapper.predict(original+3.)
    assert np.max(abs(fit['driver']-original)) > .001
    # A reference-period processed proxy is constrained through its operator;
    # adding a constant is a data inconsistency, not a raw-driver DC shift.
    assert np.max(abs(shifted['driver']-fit['driver']-3.)) > 1.
    design = mapper.map_features(features)
    np.testing.assert_allclose(mapper.hb_offset+design.sum(axis=2), fit['prediction'][:, 1:], atol=1e-14)
    features[-1] = np.nan
    np.testing.assert_array_equal(mapper.map_features(features), design)
    original[-1] = 1e30
    np.testing.assert_array_equal(mapper.predict(original)['prediction'], fit['prediction'])


def test_tak_screen_small_signal_prediction_matches_same_discrete_forward():
    n, dt = 45, .1
    p = BalloonParameters()
    mapper = build_tak_eeg_conditional_map(p, n, dt, mean_operator=reference_operator(n),
        sd=[.025, .03, .02])
    proxy = .0002*np.sin(np.arange(n)*.15)
    linear = mapper.predict(proxy)
    nonlinear = nonlinear_driver_forward(linear['driver'], linear['initial_state'], p, dt,
        hemodynamics='tak', viscoelastic_s=2., derivative=False)
    prediction = (reference_operator(n)@nonlinear['canonical_prediction'].ravel()).reshape(n, 3)
    assert np.max(abs(linear['prediction']-prediction)) < 2e-8


def test_anchored_ridge_retains_exact_pc1_and_has_no_output_normalization():
    rng = np.random.default_rng(31)
    design = rng.normal(size=(80, 2, 5))
    offset = rng.normal(size=(80, 2))*.1
    target = offset+design.sum(axis=2)
    result = fit_anchored_ridge([design], [target], sd=[.7, 1.3],
        anchor=np.ones(5), ridge=100., offsets=[offset])
    np.testing.assert_allclose(result['coefficients'], np.ones(5), atol=2e-16)
    assert not result['intercept'] and not result['output_renormalization']
    true = np.array([2., -1., .5, 0., 3.])
    result = fit_anchored_ridge([design], [design@true], sd=[1., 1.],
        anchor=np.zeros(5), ridge=0.)
    np.testing.assert_allclose(result['coefficients'], true, atol=4e-15)
    # Unit changes shared by design and target preserve the readout.
    changed = fit_anchored_ridge([design*7], [design@true*7], sd=[1., 1.],
        anchor=np.zeros(5), ridge=.2)
    original = fit_anchored_ridge([design], [design@true], sd=[1., 1.],
        anchor=np.zeros(5), ridge=.2)
    np.testing.assert_allclose(changed['coefficients'], original['coefficients'], atol=3e-15)


def test_ridge_masks_and_group_balancing_retain_full_record_structure():
    rng = np.random.default_rng(11)
    design = rng.normal(size=(20, 2, 5))
    target = rng.normal(size=(20, 2))
    mask = np.ones((20, 2), bool)
    mask[6:10] = False
    kwargs = dict(sd=[1., 1.], anchor=np.ones(5), ridge=.1, masks=[mask])
    first = fit_anchored_ridge([design], [target], **kwargs)
    target[~mask] = np.nan
    design[~mask] = np.inf
    second = fit_anchored_ridge([design], [target], **kwargs)
    np.testing.assert_array_equal(first['coefficients'], second['coefficients'])
    # Duplicating a subject's records with inverse-record weights is neutral.
    plain = fit_anchored_ridge([design, design], [target, target*2],
        **dict(kwargs, masks=[mask, mask]))
    balanced = fit_anchored_ridge([design]*3, [target, target*2, target*2],
        **dict(kwargs, masks=[mask]*3), record_weights=[1., .5, .5])
    np.testing.assert_allclose(plain['coefficients'], balanced['coefficients'], atol=1e-15)



def test_fixed_fir_lags_do_not_wrap_or_bridge_records_and_readout_keeps_reference():
    rng = np.random.default_rng(17)
    feature = rng.normal(size=(90, 1))
    design, valid = fixed_lag_fir_design(feature, [0, 3, 7])
    assert not valid[:7].any() and valid[7:].all()
    np.testing.assert_array_equal(design[7:, 1], feature[4:-3, 0])
    # The suite's explicitly declared zero-history convention allows the same
    # reference operator as Hb. The lag helper itself never fills missing data.
    padded, padded_valid = fixed_lag_fir_design(np.r_[np.zeros(7), feature[:, 0]], [0, 3, 7])
    assert padded_valid[7:].all()
    reference = np.eye(len(feature))
    reference[:, :5] -= 1/5
    processed = reference@padded[7:]
    block = np.zeros((len(feature), 2, 6))
    block[:, 0, :3], block[:, 1, 3:] = processed, processed
    true = np.array([1., -.2, .5, .3, 1., -.1])
    target = block@true
    model = fit_anchored_ridge([block], [target], sd=[1., 1.], anchor=np.zeros(6), ridge=1e-10)
    prediction = block@model['coefficients']
    np.testing.assert_allclose(prediction, target, atol=5e-10)
    np.testing.assert_allclose(prediction[:5].mean(axis=0), [0., 0.], atol=1e-16)
    feature[20] = np.nan
    _, missing = fixed_lag_fir_design(feature, [0, 3, 7])
    assert not missing[[20, 23, 27]].any()


def test_residual_statistics_preserve_gaps_sign_spectral_units_and_frozen_scale():
    dt, n = .1, 1000
    t = np.arange(n)*dt
    wave = np.sin(2*np.pi*.1*t)
    residual = np.column_stack((wave, -.5*wave))
    mask = np.ones(n, bool)
    mask[200:300] = False
    details = residual_diagnostics(residual, dt, scale=[2., 1.], mask=mask,
        lags_s=[1., 5.], regressors=wave)
    assert details['residual_sign'] == 'target_minus_prediction'
    np.testing.assert_allclose(details['normalized_rms'], [np.sqrt(.5)/2]*2, atol=1e-14)
    assert details['component_correlation'][0, 1] == pytest.approx(-1.)
    assert details['hbr_on_hbo_signed_projection'] == pytest.approx(-.5)
    assert details['hbr_to_hbo_rms_ratio'] == pytest.approx(.5)
    np.testing.assert_array_equal(details['lag_pair_count'][:, 0], [880, 800])
    assert details['spectral_band_power'][0, 1] > .48
    assert details['lag_correlation'][1, 0] == pytest.approx(-1.)
    residual[~mask] = np.nan
    hidden = residual_diagnostics(residual, dt, scale=[2., 1.], mask=mask)
    np.testing.assert_array_equal(details['rms'], hidden['rms'])


def test_invalid_contracts_fail_before_any_fitting():
    with pytest.raises(ValueError, match='initial_state_mode'):
        build_tak_eeg_conditional_map(BalloonParameters(), 10, .1, initial_state_mode='free')
    with pytest.raises(ValueError, match='eeg_visible'):
        build_tak_eeg_conditional_map(BalloonParameters(), 10, .1, eeg_visible=np.ones((10, 3), bool))
    with pytest.raises(ValueError, match='curvature_breaks'):
        build_tak_eeg_conditional_map(BalloonParameters(), 10, .1, curvature_breaks=[2])
    with pytest.raises(ValueError, match='sample lags'):
        fixed_lag_fir_design(np.ones((20, 1)), [-1, 0])
