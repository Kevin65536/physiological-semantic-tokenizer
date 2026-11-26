"""EEG band histories for the four-stream readers."""

import numpy as np


def band_histories(wave, bands):
    if wave.ndim != 4 or wave.shape[1:] != (4, 15, 400) or not np.isfinite(wave).all():
        raise ValueError("expected complete 4-site 200 Hz two-second voltage patches")
    z = (wave - wave.mean(-1, keepdims=True)) * np.hanning(400)
    power = np.abs(np.fft.rfft(z, axis=-1)) ** 2
    hz = np.fft.rfftfreq(400, 1 / 200)
    return np.stack(
        [np.log(np.maximum(power[..., (hz >= lo) & (hz < hi)].sum(-1), 1e-12)) for lo, hi in bands], axis=-1
    )
