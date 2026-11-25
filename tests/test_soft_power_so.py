from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.dual_component_tokenizer import voltage_power_cap_bank
from src.data.soft_power_so import GainObjective, final_power, fit_soft_channel


def test_soft_power_directional_gradient_and_bounded_voltage():
    rng = np.random.default_rng(17)
    x = rng.normal(size=2000)
    bank = voltage_power_cap_bank(x[:, None])[:, :, 0]
    ref = np.maximum(final_power(x).T * .15, 1e-8)
    obj = GainObjective(bank, ref, .1, 2., 1.)
    p = rng.uniform(.4, .9, 6 * len(obj.knots))
    direction = rng.normal(size=len(p))
    step = 1e-6
    approx = (obj.evaluate(p+step*direction)[0]-obj.evaluate(p-step*direction)[0])/(2*step)
    np.testing.assert_allclose(obj.evaluate(p)[1] @ direction, approx, rtol=1e-5)
    np.testing.assert_allclose(obj.voltage(np.ones_like(p)), x, atol=1e-14)
    s, gains, diag = fit_soft_channel(bank, ref, 1., .1, 2.)
    assert np.all((gains >= 0) & (gains <= 1)) and diag['final'] < diag['initial']
    np.testing.assert_allclose(s + (x-s), x, atol=1e-14)
    with pytest.raises(ValueError):
        GainObjective(bank, ref[1:], .1, 2., 1.)


def test_soft_pilot_contract():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'experiments/configs/physiology_semantic_tokenizer/soft_power_so_pilot_v1.yaml').read_text())
    assert cfg['arms'] == ['v3', 'soft_ssm', 'soft_fixed']
    assert cfg['variants'] == ['baseline', 'ocular1', 'neural10', 'neural35']
    assert cfg['seed'] == 20261121
