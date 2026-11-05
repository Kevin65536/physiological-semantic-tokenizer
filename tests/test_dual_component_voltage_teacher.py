"""Numerical and pre-read boundaries for the voltage target pilot, no local data."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml

from src.data.dual_component_tokenizer import (
    voltage_stft, voltage_istft, fit_voltage_noise_model, voltage_wiener_target,
    paired_voltage_targets, fuse_voltage_power, voltage_teacher_plan,
)


@pytest.fixture
def spec():
    path = Path(__file__).resolve().parents[1] / 'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_teacher_v1.yaml'
    return yaml.safe_load(path.read_text())['voltage_teacher']


def model(channels=3):
    return dict(transfer=np.zeros((2, channels)), correlation=np.tile(np.eye(channels), (101, 1, 1)),
        spectral_floor=np.ones((101, channels))*1e-8, band_floor=np.ones((channels, 5))*1e-8)


def test_full_spectrum_round_trip_and_absent_evidence_preserves_voltage(spec):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(18000, 3)) + np.sin(2*np.pi*80*np.arange(18000)/200)[:, None]
    _, _, z = voltage_stft(x, spec)
    assert z.shape == (361, 101, 3)
    np.testing.assert_allclose(voltage_istft(z, len(x), spec), x, atol=2e-14)
    a = rng.normal(size=(2, 15)); phys = np.ones((900, 2))*40
    result = voltage_wiener_target(x, np.zeros((18000, 2)), model(), spec,
        projection=a, physical=phys, physical_variance=np.ones(2))
    np.testing.assert_allclose(result['s'], x, atol=2e-14)
    assert result['closure_max_abs'] < 1e-14


def test_known_spatial_nuisance_reduction_and_polarity(spec):
    rng = np.random.default_rng(23); t = np.arange(18000)/200
    clean = np.column_stack([np.sin(2*np.pi*f*t) for f in (9, 12, 18)])
    eog = np.column_stack([np.sin(2*np.pi*2*t), np.cos(2*np.pi*3*t)])
    m = model(); m['transfer'] = np.array([[3., 2., 1.], [.8, -.3, .7]])
    x = clean + eog @ m['transfer']
    result = voltage_wiener_target(x, eog, m, spec)
    assert np.mean((result['s'][1000:-1000]-clean[1000:-1000])**2) < .15*np.mean((x-clean)**2)
    flipped = voltage_wiener_target(-x, -eog, m, spec)
    np.testing.assert_allclose(flipped['s'], -result['s'], atol=1e-12)
    np.testing.assert_allclose(result['s']+result['o'], x, atol=1e-14)


def test_pairing_is_exact_but_does_not_claim_teacher_robustness():
    rng = np.random.default_rng(7); x, s, n = rng.normal(size=(3, 100, 4))
    result = paired_voltage_targets(x, s, n)
    np.testing.assert_array_equal(result['s'], s)
    np.testing.assert_allclose(result['o']-(x-s), n, atol=5e-16)
    np.testing.assert_allclose(result['s']+result['o'], result['x'], atol=1e-15)


def test_low_rank_power_fusion_respects_nullspace_and_uncertainty(spec):
    rng = np.random.default_rng(3); ell = rng.normal(size=(10, 15))
    a = np.zeros((2, 15)); a[0, 0] = 1; a[1, 1] = 1
    target = np.zeros((10, 2)); noise = np.zeros_like(ell)
    delta, _ = fuse_voltage_power(ell, noise, a, target, np.ones(2), spec)
    assert np.all(delta[:, 2:] == 0)
    weak, _ = fuse_voltage_power(ell, noise, a, target, np.ones(2)*1e9, spec)
    assert np.linalg.norm(weak) < 1e-7*np.linalg.norm(delta)


def test_noise_calibration_rejects_validation_before_array_access(spec):
    with pytest.raises(ValueError, match='training subjects only'):
        fit_voltage_noise_model([dict(split='validation')], spec)


def test_plan_rejects_test_subject_and_preserves_failed_denominator(spec):
    s = deepcopy(spec); s['train_subjects'] = ['train01']; s['validation_subjects'] = ['val01']
    cfg = dict(voltage_teacher=s, data=dict(protected_data='forbidden'))
    retained = dict(subjects=dict(simultaneous_eeg_nirs=dict(train=['train01'], validation=['val01'], test=['test01'])), rows=[])
    result = voltage_teacher_plan(cfg, retained)
    assert result['planned_windows'] == 4 and len(result['excluded']) == 2
    cfg['voltage_teacher']['validation_subjects'] = ['test01']
    with pytest.raises(ValueError, match='parent development partition'):
        voltage_teacher_plan(cfg, retained)


def test_intervention_metric_distinguishes_artifact_and_neural_changes():
    from src.analysis.dual_component_tokenizer import voltage_teacher_delta_metrics
    rng = np.random.default_rng(8); base = rng.normal(size=(18000, 2)); n = rng.normal(size=base.shape)
    passthrough = voltage_teacher_delta_metrics(base, base+n, n, 'ocular')
    assert passthrough['delta_error_NMSE'] == pytest.approx(1.)
    assert passthrough['signed_projection_S'] == pytest.approx(1.)
    retained = voltage_teacher_delta_metrics(base, base+n, n, 'neural_alpha')
    assert retained['delta_error_NMSE'] < 1e-28
    removed = voltage_teacher_delta_metrics(base, base, n, 'ocular')
    assert removed['delta_error_NMSE'] == 0.
    assert removed['signed_projection_O'] == 1.
    inverted = voltage_teacher_delta_metrics(base, base-n, n, 'ocular')
    assert inverted['signed_projection_S'] == pytest.approx(-1.)
    assert inverted['signed_projection_O'] == pytest.approx(2.)


def test_voltage_teacher_rejects_test_before_native_reader(monkeypatch):
    from src.data import dual_component_tokenizer as data
    monkeypatch.setattr(data, 'load_native_eeg_record', lambda *args: pytest.fail('native signal reader reached'))
    entries=[dict(rows=[dict(split='test',dataset='simultaneous_eeg_nirs')])]
    with pytest.raises(ValueError, match='public development panel'):
        list(data.prepare_voltage_teacher_record('.',{},entries,{},None))
