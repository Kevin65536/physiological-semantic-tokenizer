import numpy as np
from scipy import signal
from experiments.scripts.render_continuous_so_phase_atlas import (
    spectral_pair,
    alpha_pair,
    TIME,
    INTERVAL,
    FREQ,
)


def test_new_minus_old_phase_power_and_low_energy_mask():
    old = np.sin(2 * np.pi * 10 * TIME)
    new = 2 * np.sin(2 * np.pi * 10 * TIME + np.pi / 3)
    pa, pb, phase, coh, valid = spectral_pair(old[INTERVAL], new[INTERVAL])
    freq, expected = signal.welch(old[INTERVAL], fs=200, nperseg=400, noverlap=200)
    np.testing.assert_array_equal(freq, FREQ)
    np.testing.assert_allclose(pa, expected, atol=1e-12)
    at = FREQ == 10
    assert valid[at] and not valid[FREQ == 30]
    np.testing.assert_allclose(pb[at] / pa[at], 4)
    np.testing.assert_allclose(phase[at], 60, atol=1e-8)
    _, delta, mask = alpha_pair(old, new)
    assert mask[INTERVAL].all()
    np.testing.assert_allclose(np.median(delta[INTERVAL]), 60, atol=0.01)
