"""Synthetic contracts for train-frozen gain and masked HbR completion."""
from pathlib import Path
import json
import numpy as np
import pytest
from experiments.scripts.analyze_hbo_hbr_relationship import (
    predictive_config, predictive_checks, predictive_fit, predictive_gain,
    predictive_apply, synthetic_adaptation_curves,
)

CONFIG = Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_predictive_separation_v1.yaml'


def test_analytic_transfer_gain_and_hidden_value_invariance():
    cfg = predictive_config(CONFIG)
    checks = predictive_checks(cfg)
    assert checks['analytic_forward_relative_error'] < .001
    assert checks['hidden_perturbation_max_error'] == 0


def test_recover_physiology_with_fixed_known_gain_from_independent_analytic_generator():
    cfg = predictive_config(CONFIG)
    y = synthetic_adaptation_curves([3., .4, .32, .32], np.linspace(.1, 6, 12))
    y[:, 1] *= 1.7
    fit = predictive_fit(y, cfg, gain=1.7)
    assert abs(fit['tau']-3) < .03
    assert abs(fit['eta']-.4) < .01


def test_gain_and_physiology_unchanged_by_common_scale_and_offsets():
    cfg = predictive_config(CONFIG)
    y = synthetic_adaptation_curves([2., .35, .32, .32], np.linspace(.1, 6, 8))
    changed = y*3+np.array([2., -4.])[None, :, None]
    assert abs(predictive_gain(y, cfg)-predictive_gain(changed, cfg)) < 1e-10


def test_invalid_tensor_and_contract_fail_before_data_access(tmp_path):
    cfg = predictive_config(CONFIG)
    assert json.loads(json.dumps(cfg)) == cfg
    with pytest.raises(ValueError):
        predictive_apply(np.zeros((2, 300)), 2., .35, 1., cfg)
    path = tmp_path/'invalid.yaml'
    path.write_text(CONFIG.read_text().replace('samples: 300', 'samples: 299'))
    with pytest.raises(ValueError):
        predictive_config(path)
