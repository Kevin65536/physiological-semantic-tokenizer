"""Array-only checks for frozen spatial observation coordinates and readouts."""

import inspect

import numpy as np
import pytest

from src.inference.observation_baselines import (
    fit_spatial_coordinates,
    fit_spatial_eeg_loadings,
    spatial_observation_design,
    transform_spatial_coordinates,
)


@pytest.fixture
def montage():
    eeg = np.array([[-1.5, .2], [-.4, .4], [.2, 1.], [1.1, .1], [2., -.3]])
    hb = np.array([[0., 0.], [.5, .3], [1.4, -.2], [2.5, .2]])
    return eeg, hb


@pytest.mark.parametrize('modes', [1, 2])
def test_sham_preserves_capacity_spectrum_and_declared_target(montage, modes):
    eeg, hb = montage
    true = spatial_observation_design(eeg, hb, modes, kernel_width=.8)
    sham = spatial_observation_design(eeg, hb, modes, geometry='sham', seed=13,
                                      kernel_width=.8)
    repeated = spatial_observation_design(eeg, hb, modes, geometry='sham', seed=13,
                                          kernel_width=.8)
    assert true['learned_band_coefficients'] == sham['learned_band_coefficients'] == 5*modes
    assert true['ranks'] == sham['ranks'] == dict(eeg=modes, hb=modes)
    assert sham['anchors'] == true['anchors']
    np.testing.assert_array_equal(sham['anchor_positions'], true['anchor_positions'])
    np.testing.assert_array_equal(sham['spatial_eeg'],
                                  true['spatial_eeg'][sham['permutation']['eeg']])
    np.testing.assert_array_equal(sham['hb_loadings'],
                                  true['hb_loadings'][sham['permutation']['hb']])
    assert sham['permutation']['hb'][0] == 0
    assert sham['eeg_changed']
    assert sham['hb_changed'] is (modes == 2)
    for key in ('spatial_eeg', 'hb_loadings'):
        np.testing.assert_allclose(np.linalg.svd(sham[key], compute_uv=False),
                                   np.linalg.svd(true[key], compute_uv=False), atol=1e-15)
        np.testing.assert_array_equal(sham[key], repeated[key])
    np.testing.assert_allclose(np.linalg.norm(true['spatial_eeg'], axis=0), 1.)
    np.testing.assert_allclose(true['hb_loadings'].sum(axis=1), 1.)
    assert np.all(true['spatial_eeg'] > 0) and np.all(true['hb_loadings'] > 0)
    if modes == 1:
        np.testing.assert_array_equal(true['hb_loadings'], np.ones((len(hb), 1)))
    else:
        assert true['anchors'] == [0, 3]


def test_true_design_tracks_signal_order_and_freezes_derived_width(montage):
    eeg, hb = montage
    true = spatial_observation_design(eeg, hb, 2)
    eeg_order, hb_order = [3, 0, 4, 1, 2], [0, 3, 1, 2]
    reordered = spatial_observation_design(eeg[eeg_order], hb[hb_order], 2)
    np.testing.assert_allclose(reordered['spatial_eeg'], true['spatial_eeg'][eeg_order])
    np.testing.assert_allclose(reordered['hb_loadings'], true['hb_loadings'][hb_order])
    assert reordered['kernel_width'] == true['kernel_width']
    assert reordered['anchors'] == [0, 1]


def test_uniform_is_one_core_and_two_hb_channels_have_no_donor_sham(montage):
    eeg, hb = montage
    uniform = spatial_observation_design(eeg, hb, geometry='uniform')
    np.testing.assert_array_equal(uniform['spatial_eeg'], np.ones((len(eeg), 1))/np.sqrt(len(eeg)))
    np.testing.assert_array_equal(uniform['hb_loadings'], np.ones((len(hb), 1)))
    with pytest.raises(ValueError, match='only for K=1'):
        spatial_observation_design(eeg, hb, 2, geometry='uniform')
    sham = spatial_observation_design(eeg, hb[:2], 2, geometry='sham')
    np.testing.assert_array_equal(sham['permutation']['hb'], [0, 1])
    assert sham['hb_changed'] is False and sham['eeg_changed'] is True


def test_degenerate_montage_and_vacuous_sham_fail_explicitly(montage):
    eeg, hb = montage
    with pytest.raises(ValueError, match='distinct Hb anchors'):
        spatial_observation_design(eeg, np.zeros_like(hb), 2)
    with pytest.raises(ValueError, match='rank K'):
        spatial_observation_design(np.zeros_like(eeg), hb, 2)
    with pytest.raises(ValueError, match='nontrivial EEG'):
        spatial_observation_design(np.ones_like(eeg), hb, geometry='sham')
    with pytest.raises(ValueError, match='positive finite'):
        spatial_observation_design(eeg, hb, kernel_width=0.)
    with pytest.raises(ValueError, match='finite EEG'):
        spatial_observation_design(eeg[:, :1], hb)


def _training_pairs(seed=81):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(17, 4, 5))+.5, rng.normal(size=(17, 3, 2))*.2),
            (rng.normal(size=(23, 4, 5))*3.+.5, rng.normal(size=(23, 3, 2))*.6)]


def test_coordinates_are_pooled_training_rms_without_record_centering():
    chunks = _training_pairs()
    coordinate = fit_spatial_coordinates(iter(chunks), training_ids=['train_a', 'train_b'])
    eeg = np.concatenate([pair[0] for pair in chunks])
    hb = np.concatenate([pair[1] for pair in chunks])
    transformed_eeg, transformed_hb = transform_spatial_coordinates(eeg, hb, coordinate)
    expected_band_scale = np.sqrt(np.mean(eeg**2, axis=(0, 1)))/.025
    np.testing.assert_allclose(coordinate['eeg_scale'], np.broadcast_to(expected_band_scale, eeg.shape[1:]))
    assert coordinate['hb_scale'] == pytest.approx(np.sqrt(np.mean(hb**2))/.025)
    np.testing.assert_allclose(np.sqrt(np.mean(transformed_eeg.reshape(eeg.shape)**2, axis=(0, 1))), .025)
    assert np.sqrt(np.mean(transformed_hb**2)) == pytest.approx(.025)
    expected = np.column_stack((transformed_eeg, transformed_hb.reshape(len(hb), -1)))
    np.testing.assert_allclose(coordinate['sd'], expected.std(axis=0))
    assert np.linalg.norm(transformed_eeg.mean(axis=0)) > .001
    assert coordinate['training_count'] == 2 and coordinate['training_rows'] == 40
    assert coordinate['training_ids'] == ['train_a', 'train_b']
    assert coordinate['centering'] == 'none_already_referenced_inputs'
    # Repartitioning the same rows cannot change the coordinate gauge.
    pooled = fit_spatial_coordinates([(eeg, hb)])
    np.testing.assert_allclose(pooled['eeg_scale'], coordinate['eeg_scale'])
    assert pooled['hb_scale'] == coordinate['hb_scale']


def test_band_shared_scale_preserves_known_spatial_rank_one_and_true_geometry(montage):
    eeg_positions, hb_positions = montage
    true = spatial_observation_design(eeg_positions, hb_positions, kernel_width=.6)
    uniform = spatial_observation_design(eeg_positions, hb_positions, geometry='uniform')
    spatial = true['spatial_eeg'][:, 0]
    bands = np.array([1., -.6, .2, -.8, .4])
    rng = np.random.default_rng(31)
    driver = rng.normal(size=(100, 1))*.4
    eeg = driver[:, :, None]*spatial[None, :, None]*bands[None, None, :]
    hb = rng.normal(size=(100, len(hb_positions), 2))*.02
    coordinate = fit_spatial_coordinates([(eeg, hb)])
    transformed, _ = transform_spatial_coordinates(eeg, hb, coordinate)
    fitted_true = fit_spatial_eeg_loadings([transformed], [driver], true['spatial_eeg'], ridge=0.)
    fitted_uniform = fit_spatial_eeg_loadings([transformed], [driver], uniform['spatial_eeg'], ridge=0.)
    # A per-channel RMS would erase spatial and make a uniform kernel fit.
    # Sharing one scale within each band preserves the declared factorization.
    expected_bands = bands/coordinate['eeg_scale'][0]
    np.testing.assert_allclose(fitted_true['band_coefficients'][:, 0], expected_bands, atol=1e-15)
    np.testing.assert_allclose(fitted_true['eeg_loadings'][:, 0].reshape(len(spatial), 5),
                               spatial[:, None]*expected_bands[None, :], atol=1e-15)
    true_error = np.mean((driver@fitted_true['eeg_loadings'].T-transformed)**2)
    uniform_error = np.mean((driver@fitted_uniform['eeg_loadings'].T-transformed)**2)
    assert true_error < 1e-28 and uniform_error > 1e-5


def test_common_hb_scale_preserves_species_ratio_total_and_between_record_amplitude():
    eeg = np.arange(1., 121.).reshape(6, 4, 5)
    hb = np.arange(1., 37.).reshape(6, 3, 2)
    hb[..., 1] *= -.3
    coordinate = fit_spatial_coordinates([(eeg, hb), (2.*eeg, 2.*hb)])
    first_eeg, first_hb = transform_spatial_coordinates(eeg, hb, coordinate)
    second_eeg, second_hb = transform_spatial_coordinates(2.*eeg, 2.*hb, coordinate)
    np.testing.assert_allclose(first_hb[..., 0]/first_hb[..., 1], hb[..., 0]/hb[..., 1])
    np.testing.assert_allclose(first_hb.sum(axis=-1), hb.sum(axis=-1)/coordinate['hb_scale'])
    np.testing.assert_allclose(second_eeg, 2.*first_eeg)
    np.testing.assert_allclose(second_hb, 2.*first_hb)


def test_frozen_transform_hidden_poison_cannot_change_any_visible_coordinate():
    chunks = _training_pairs()
    coordinate = fit_spatial_coordinates(chunks)
    eeg, hb = (value.copy() for value in chunks[0])
    before_eeg, before_hb = transform_spatial_coordinates(eeg, hb, coordinate)
    visible = np.ones_like(hb, dtype=bool)
    visible[4:11, 0] = False
    hb[~visible] = 1e20
    after_eeg, after_hb = transform_spatial_coordinates(eeg, hb, coordinate)
    np.testing.assert_array_equal(after_eeg, before_eeg)
    np.testing.assert_array_equal(after_hb[visible], before_hb[visible])
    hb[~visible] = np.nan
    _, missing_hb = transform_spatial_coordinates(eeg, hb, coordinate)
    assert np.isnan(missing_hb[~visible]).all()
    np.testing.assert_array_equal(missing_hb[visible], before_hb[visible])
    np.testing.assert_array_equal(transform_spatial_coordinates(eeg.reshape(len(eeg), -1), hb, coordinate)[0],
                                  before_eeg)


def test_coordinate_fitting_has_no_evaluation_array_argument_and_rejects_missing_training():
    assert set(inspect.signature(fit_spatial_coordinates).parameters) == {
        'chunks', 'training_ids', 'target_rms'}
    with pytest.raises(ValueError, match='at least one'):
        fit_spatial_coordinates([])
    chunks = _training_pairs()
    with pytest.raises(ValueError, match='training_ids'):
        fit_spatial_coordinates(chunks, training_ids=['wrong_length'])
    eeg, hb = chunks[0]
    with pytest.raises(ValueError, match='positive finite observation scales'):
        fit_spatial_coordinates([(eeg*0., hb)])
    missing = hb.copy()
    missing[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite training'):
        fit_spatial_coordinates([(eeg, missing)])


@pytest.mark.parametrize('modes', [1, 2])
def test_signed_band_loading_recovery_keeps_spatial_factor_and_driver_gauge(montage, modes):
    eeg, hb = montage
    spatial = spatial_observation_design(eeg, hb, modes, kernel_width=.8)['spatial_eeg']
    rng = np.random.default_rng(42)
    drivers = [rng.normal(size=(37, modes))*.4, rng.normal(size=(29, modes))*.4]
    bands = np.array([[1., .2], [-.8, .5], [.3, -.9], [-.4, -.7], [.9, 1.1]])[:, :modes]
    loading = (spatial[:, None, :]*bands[None, :, :]).reshape(len(eeg)*5, modes)
    observations = [(driver@loading.T).reshape(len(driver), len(eeg), 5) for driver in drivers]
    before = [driver.copy() for driver in drivers]
    fitted = fit_spatial_eeg_loadings([observations[0], observations[1].reshape(29, -1)],
                                     drivers, spatial, ridge=0., training_ids=['a', 'b'])
    np.testing.assert_allclose(fitted['eeg_loadings'], loading, atol=1e-14)
    np.testing.assert_allclose(fitted['band_coefficients'], bands, atol=1e-14)
    assert fitted['rank'] == fitted['design_rank'] == modes
    assert fitted['learned_coefficients'] == 5*modes
    assert fitted['training_count'] == 2 and fitted['training_rows'] == 66
    assert fitted['training_ids'] == ['a', 'b']
    for after, original in zip(drivers, before):
        np.testing.assert_array_equal(after, original)
    # Coefficients change inversely with a changed reference gauge; helpers
    # do not secretly renormalize the fitted loading or supplied drivers.
    changed = fit_spatial_eeg_loadings(observations, [3.*r for r in drivers], spatial, ridge=0.)
    np.testing.assert_allclose(changed['eeg_loadings'], loading/3., atol=1e-14)


@pytest.mark.parametrize('modes', [1, 2])
@pytest.mark.parametrize('unit_factor', [.001, 3., 1000.])
def test_relative_gram_ridge_preserves_prediction_under_driver_unit_changes(montage, modes, unit_factor):
    eeg, hb = montage
    spatial = spatial_observation_design(eeg, hb, modes, kernel_width=.8)['spatial_eeg']
    rng = np.random.default_rng(19)
    driver = rng.normal(size=(80, modes))*.025
    band_coefficients = rng.normal(size=(5, modes))
    loading = (spatial[:, None, :]*band_coefficients[None, :, :]).reshape(len(eeg)*5, modes)
    features = driver@loading.T+rng.normal(size=(80, len(eeg)*5))*.0001
    fitted = fit_spatial_eeg_loadings([features], [driver], spatial, ridge=.1)
    changed = fit_spatial_eeg_loadings([features], [unit_factor*driver], spatial, ridge=.1)
    np.testing.assert_allclose(changed['eeg_loadings'], fitted['eeg_loadings']/unit_factor,
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(unit_factor*driver@changed['eeg_loadings'].T,
                               driver@fitted['eeg_loadings'].T, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fitted['effective_ridge'], .1*fitted['gram_trace']/modes)
    np.testing.assert_allclose(changed['effective_ridge'], unit_factor**2*fitted['effective_ridge'])
    assert len(fitted['effective_ridge']) == 5
    assert fitted['ridge_definition'] == 'penalty_b = ridge * trace(X_b.T @ X_b / n_rows) / K'


def test_band_only_capacity_cannot_fit_away_wrong_geometry(montage):
    eeg, hb = montage
    true = spatial_observation_design(eeg, hb, kernel_width=.6)
    sham = spatial_observation_design(eeg, hb, geometry='sham', seed=4, kernel_width=.6)
    driver = np.random.default_rng(12).normal(size=(80, 1))
    coefficients = np.array([1., -.6, .2, -.8, .4])
    loading = (true['spatial_eeg'][:, 0, None]*coefficients).ravel()[:, None]
    features = driver@loading.T
    fitted_true = fit_spatial_eeg_loadings([features], [driver], true['spatial_eeg'], ridge=0.)
    fitted_sham = fit_spatial_eeg_loadings([features], [driver], sham['spatial_eeg'], ridge=0.)
    correct_error = np.mean((driver@fitted_true['eeg_loadings'].T-features)**2)
    sham_error = np.mean((driver@fitted_sham['eeg_loadings'].T-features)**2)
    assert correct_error < 1e-28 and sham_error > .01
    assert fitted_true['learned_coefficients'] == fitted_sham['learned_coefficients'] == 5
    np.testing.assert_allclose(fitted_sham['eeg_loadings'].reshape(len(eeg), 5)
                               /sham['spatial_eeg'],
                               np.broadcast_to(fitted_sham['band_coefficients'][:, 0], (len(eeg), 5)))


def test_loading_fit_rejects_unidentifiable_or_zero_loading(montage):
    eeg, hb = montage
    spatial = spatial_observation_design(eeg, hb, 2)['spatial_eeg']
    features = np.ones((10, len(eeg), 5))
    with pytest.raises(ValueError, match='rank deficient'):
        fit_spatial_eeg_loadings([features], [np.zeros((10, 2))], spatial)
    driver = np.random.default_rng(91).normal(size=(10, 2))
    with pytest.raises(ValueError, match='calibration unavailable'):
        fit_spatial_eeg_loadings([features], [driver*1e-170], spatial)
    with pytest.raises(ValueError, match='near-zero'):
        fit_spatial_eeg_loadings([features*0.], [driver], spatial)
    with pytest.raises(ValueError, match='paired training'):
        fit_spatial_eeg_loadings([features], [], spatial)
    with pytest.raises(ValueError, match='training_ids'):
        fit_spatial_eeg_loadings([features], [driver], spatial, training_ids=['a', 'b'])
    invalid = features.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite paired training'):
        fit_spatial_eeg_loadings([invalid], [driver], spatial)
