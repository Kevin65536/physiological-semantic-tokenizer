"""Check only the new paired measurement logic on synthetic voltage."""
import numpy as np
import pytest

from experiments.scripts import evaluate_unified_eeg_so_pilot as pilot
from src.data.unified_physiology import _bandpass, _resample


@pytest.mark.parametrize('rate,start', [(200, 41), (500, 38), (1000, 44)])
def test_addition_uses_record_grid_and_preserves_filter_loss(rate, start):
    t = np.arange(90*rate)/rate
    window = (t>=30) & (t<50)
    delta = (window*np.sin(np.pi*np.clip((t-30)/20,0,1))**2*np.cos(2*np.pi*55*t))[:,None]
    padded = np.pad(delta, ((start, 3*rate), (0,0)))
    full = _resample(_bandpass(padded,rate,(1,45)),rate,200)
    lo = round(start*200/rate)
    observed = pilot.filter_delta(delta,rate,start)
    np.testing.assert_allclose(observed,full[lo:lo+18000],atol=1e-10,rtol=1e-8)
    assert pilot.rms(observed) < .5*pilot.rms(delta)


def test_orthogonal_leakage_and_signed_gain_measure_different_changes():
    delta=np.array([-1.,1.,-1.,1.])
    s_change=np.array([1.,1.,-1.,-1.])
    scores=pilot.change_score(s_change,delta)
    assert np.isclose(scores['leak'],1) and np.isclose(scores['gain'],1)
    assert np.isclose(scores['o_corr'],1/np.sqrt(2))
    assert np.isclose(scores['preservation_error'],np.sqrt(2))
