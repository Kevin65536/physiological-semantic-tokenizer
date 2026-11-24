"""Regression checks for large-K pattern readouts and deterministic atlas members."""
import importlib.util
from pathlib import Path

import numpy as np

from src.analysis.tokenizer_pattern_audit import PatternAudit, SparsePatternAudit

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/scripts/audit_joint_tokenizer_patterns.py'
spec = importlib.util.spec_from_file_location('joint_patterns', SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_sparse_categorical_readouts_match_existing_dense_readouts():
    fields = {'wave': (0, 4), 'slope': (4, 5)}
    dense = PatternAudit(7, 3, ['a'], 5, fields)
    sparse = SparsePatternAudit(7, 3, ['a'], 5, fields)
    rng = np.random.default_rng(3)
    code = rng.integers(0, 5, size=100)
    background = rng.integers(0, 2, size=100)
    values = rng.normal(size=(100, 5))
    for audit in (dense, sparse):
        audit.fit(code[:40], background[:40], values[:40])
        audit.fit(code[40:], background[40:], values[40:])
    query_code = np.array([0, 4, 6, 4])
    query_background = np.array([0, 1, 2, 2])
    for wanted, actual in zip(dense.read(query_code, query_background), sparse.read(query_code, query_background)):
        np.testing.assert_allclose(actual, wanted)
    assert len(sparse.joint.count) <= 10


def test_reservoir_priorities_are_stable_across_batch_boundaries():
    left = script.MemberReservoir(seed=7, capacity=5)
    right = script.MemberReservoir(seed=7, capacity=5)
    ids = np.arange(100) % 3
    waves = np.arange(200).reshape(100, 2)
    metadata = np.arange(400).reshape(100, 4)
    left.add('train', ids, waves, metadata)
    for first in range(0, 100, 7):
        right.add('train', ids[first:first + 7], waves[first:first + 7], metadata[first:first + 7])
    for key in left.members:
        for a, b in zip(left.members[key], right.members[key]):
            np.testing.assert_array_equal(a, b)


def test_centered_hb_shape_retains_slopes_and_removes_only_diagnostic_level():
    t = np.arange(20) / 10
    wave = np.stack([4 + 2 * t, -3 - t], axis=-1).reshape(1, 40)
    features, fields = script.pattern_attributes(wave, 'hb')
    for name, slope in [('HbO', 2), ('HbR', -1)]:
        a, b = fields[name + '_centered']
        np.testing.assert_allclose(features[:, a:b].mean(), 0, atol=1e-14)
        np.testing.assert_allclose(np.diff(features[:, a:b]) * 10, slope)
        a, b = fields[name + '_slope']
        np.testing.assert_allclose(features[:, a:b], slope)


def test_context_forward_and_backward_keep_the_same_physical_patch(tmp_path, monkeypatch):
    from src.analysis.tokenizer_pattern_audit import SparsePatternAudit
    rows = [dict(dataset='d', subject='v', record='r', parent='p', positions=['p'],
                 inputs={'score': [{'start_s': s, 'positions': [0]} for s in (0, 30, 60)]})]
    examples = np.array([[0, 0, 0], [0, 60, 0]])
    fields = {'level': (0, 1)}
    audits = {b: SparsePatternAudit(64, 1, ['d'], 1, fields) for b in 'SO'}
    for audit in audits.values():
        audit.fit(np.arange(64), np.zeros(64, dtype=int), np.arange(64)[:, None])

    def batch(rows, examples, scales, **kwargs):
        x = np.stack([(start // 2 + np.arange(15))[:, None] for _, start, _ in examples])
        return {'x': x, 'mask': np.ones_like(x, dtype=bool)}

    class Model:
        def encode(self, values):
            return {b: values[..., 0].astype(int) for b in 'SO'}

    monkeypatch.setattr(script, 'base_tokenizer_batch', batch)
    result = script.natural_context(Model(), rows, {}, examples, audits, fields, 'hb', tmp_path, {}, 4,
                                    {b: {'level': 1} for b in 'SO'})
    for b in 'SO':
        assert result[b]['complete_central_patches'] == 2
        assert result[b]['attribute_change_over_training_variance']['level'] == 0
        with np.load(tmp_path / (b + '_natural_context.npz')) as archive:
            np.testing.assert_array_equal(archive['original_ID'], archive['shifted_ID'])


def test_training_residual_readout_and_infinite_shrinkage():
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    import audit_joint_tokenizer_readout_variance as variance
    fields = {'level': (0, 1)}
    audit = SparsePatternAudit(2, 2, ['d'], 1, fields)
    ids = np.tile([0, 1], 4)
    backgrounds = np.repeat([0, 1], 4)
    values = (10 * backgrounds + 2 * ids)[:, None]
    audit.fit(ids, backgrounds, values)
    residual = variance.additive_residual(audit)
    np.testing.assert_allclose(residual[:, 0], [-1, 1])
    prediction, count = variance.readout_predictions(audit, ids, backgrounds, residual)
    np.testing.assert_array_equal(count, 2)
    np.testing.assert_allclose(prediction[1], prediction[0])
    np.testing.assert_allclose(prediction[1 + len(variance.LAMBDAS)], prediction[0])
    np.testing.assert_allclose(prediction[-1], values)
    np.testing.assert_allclose(prediction[len(variance.LAMBDAS)], values)


def test_screen_gate_does_not_generate_confirmation_verdict():
    primary = ['power_4_8', 'power_8_13', 'rms_envelope']
    branch = dict(additional_ID_gain_over_background={name: .2 for name in primary},
                  positive_subject_fraction={name: 1. for name in primary},
                  natural_context={'attribute_change_over_training_variance': {name: .01 for name in primary}},
                  validation_mass_supported=1., by_dataset_gain={'d': {name: .2 for name in primary}})
    gate = script.pattern_gate(branch, 'eeg', 'screen')
    assert gate['panel_role'] == 'screen'
    assert 'panel_positive_datasets' in gate
    assert not any(name.startswith('confirmation') for name in gate)
    confirm = script.pattern_gate(branch, 'eeg', 'confirm')
    assert 'confirmation_three_datasets_pass' in confirm
    assert not confirm['numerical_pass']
