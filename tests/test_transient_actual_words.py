"""Source-fold ancestry and original categorical information cannot be substituted."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def owner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    return importlib.import_module('audit_transient_actual_words')


def test_family_selection_freezes_waveform_rejection_and_uses_both_components(monkeypatch):
    module = owner(monkeypatch)
    def record(values, eligible=False):
        return dict(train_waveform_eligible=eligible, arms={arm: dict(metrics=dict(nmse=dict(macro=np.asarray(v)[:, None]))) for arm, v in [('legacy', [1, 1, 1]), ('transient', values)]})
    summary = dict(families=dict(high=record([.9, .7, .9]), semantic=record([.8, .65, 1.])))
    assert module.select_families(summary)['families'] == ['high']
    assert 'mechanism_only' in module.select_families(summary)['mode']
    summary['families']['semantic']['train_waveform_eligible'] = True
    assert module.select_families(summary)['families'] == ['semantic']
    summary['families']['high']['train_waveform_eligible'] = True
    assert module.select_families(summary)['families'] == ['high', 'semantic']


def test_additional_flagship_keeps_original_selected_family(monkeypatch):
    module = owner(monkeypatch)
    def record(values):
        return dict(train_waveform_eligible=False, arms={arm: dict(metrics=dict(nmse=dict(macro=np.asarray(v)[:, None]))) for arm, v in [('legacy', [1, 1, 1]), ('transient', values)]})
    summary = dict(families=dict(high=record([1, .9, .99]), semantic=record([1, .8, .98])))
    chosen = module.selected_scope(summary, dict(selection=dict(additional_family='high', additional_family_role='existing_flagship_mechanism')))
    assert chosen['primary_selected_families'] == ['semantic']
    assert chosen['families'] == ['semantic', 'high']
    assert not any(chosen['original_waveform_eligibility'].values())


def test_held_source_cannot_fit_its_own_readout_fold(monkeypatch):
    module = owner(monkeypatch)
    folds = {'study': {'a': 0, 'b': 1, 'c': 2}}
    coordinate = dict(input_identity='input', scales_identity='scale')
    meta = dict(held_fold=0, fit_subjects=['b', 'c'], positions={'study': ['Fz']}, **coordinate)
    module.check_source(meta, 'study', 0, folds, ['Fz'], coordinate)
    with pytest.raises(ValueError, match='same-fold'):
        module.check_source(dict(meta, held_fold=1, fit_subjects=['a', 'c']), 'study', 0, folds, ['Fz'], coordinate)


def test_original_raw_conditional_means_and_unseen_background_fallback(monkeypatch):
    module = owner(monkeypatch)
    person = np.repeat(np.arange(4), 2)
    target = np.tile([1., 3.], 4)
    bank = dict(fold=person//2, subject=person, valid=np.ones((8, 15), bool),
        fixed_band=np.zeros(8, int), background=np.zeros((8, 15), int),
        S_attributes=np.broadcast_to(target[:, None, None], (8, 15, 10)))
    codes = {'S': np.broadcast_to(np.tile([1, 2], 4)[:, None], (8, 15))}
    held, error, mass, usage, counts = module.raw_readout(bank, codes, 1, 'S', 3, 1)
    np.testing.assert_array_equal(held, [4, 5, 6, 7])
    np.testing.assert_array_equal(error[:, 0], 1.)
    np.testing.assert_array_equal(error[:, 1], 0.)
    np.testing.assert_array_equal(mass, 1.)
    assert counts['fit_people'] == [0, 1]
    assert len(usage) == 15
    # A held-only ID must fall back to the same fitted background mean.
    codes['S'] = codes['S'].copy();codes['S'][4:] = 0
    np.testing.assert_array_equal(module.raw_readout(bank, codes, 1, 'S', 3, 1)[1][:, 1], 1.)
