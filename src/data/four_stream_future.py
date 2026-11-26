"""Sample-aligned, bounded observations for the Simultaneous released n-back panel."""

import numpy as np

from src.analysis.single_vq_relations import wave_attribute
from src.data.homer2_preprocessing import bandpass_fnirs


def sample_limits(marker_ms, rate):
    # Exclusive stop rounded down on each native clock. Future starts round up.
    stop = int(np.floor((marker_ms + 4000) * rate / 1000 + 1e-9))
    starts = np.ceil((marker_ms / 1000 + np.array([6.0, 8.0, 10.0])) * rate - 1e-9).astype(int)
    return stop - 30 * rate, stop, starts


def event_observations(eeg, hb, eeg_marker_ms, hb_marker_ms, *, cached_hb=False):
    """EEG [time,4] uV, Hb [time,4,2] uM. Models never see target arrays."""
    e0, e1, es = sample_limits(eeg_marker_ms, 200)
    h0, h1, hs = sample_limits(hb_marker_ms, 10)
    if e0 < 0 or h0 < 0 or es[-1] + 400 > len(eeg) or hs[-1] + 20 > len(hb):
        raise ValueError("incomplete native time support")
    e, h = eeg[e0:e1], hb[h0:h1]
    ef = np.stack([eeg[t : t + 400].T for t in es], axis=1)
    hf = np.stack([hb[t : t + 20].mean(0) for t in hs], axis=1)
    if not all(np.isfinite(v).all() for v in (e, h, ef, hf)):
        raise ValueError("nonfinite measurement")
    if np.any(np.ptp(e, axis=0) == 0) or np.any(np.ptp(h, axis=0) == 0):
        raise ValueError("constant history")
    freq = np.abs(np.fft.rfft(e, axis=0)) ** 2
    freq[1:-1] *= 2
    if (freq[49 * 30 : 51 * 30 + 1].sum(0) > 0.5 * freq.sum(0)).any():
        raise ValueError("line50 history")
    if cached_hb:
        filt = h.copy()
    else:
        filt, status = bandpass_fnirs(h, sample_rate_hz=10)
        if status["status"] != "applied":
            raise ValueError("Hb filter failed")
    filt -= filt[:50].mean(0)
    voltage = e.T.reshape(4, 15, 400)
    alpha = wave_attribute(voltage, "eeg")
    return dict(
        eeg_wave=voltage.astype("float32"),
        hb_wave=filt.transpose(1, 0, 2).reshape(4, 15, 40).astype("float32"),
        hb_history=h.transpose(1, 0, 2).astype("float32"),
        eeg_alpha=alpha.astype("float32"),
        eeg_target=(wave_attribute(ef, "eeg") - alpha[:, -1:]).astype("float32"),
        hb_target=(hf - h[-20:].mean(0)[:, None]).astype("float32"),
        encoder_start=np.array([e0 / 200 - eeg_marker_ms / 1000, h0 / 10 - hb_marker_ms / 1000]),
        raw_stop=np.array(
            [e1 / 200 - eeg_marker_ms / 1000, (len(hb) if cached_hb else h1) / 10 - hb_marker_ms / 1000]
        ),
        eeg_target_start=es / 200 - eeg_marker_ms / 1000,
        hb_target_start=hs / 10 - hb_marker_ms / 1000,
    )


def lowpass_split(history, mod, cutoff):
    """Complementary FFT projection inside one 30s history, preserving the two Hb colors."""
    rate, colors = (200, 1) if mod == "eeg" else (10, 2)
    series = history.reshape(*history.shape[:-2], 30 * rate, colors)
    spec = np.fft.rfft(series, axis=-2)
    spec[..., np.fft.rfftfreq(30 * rate, 1 / rate) > cutoff, :] = 0
    smooth = np.fft.irfft(spec, n=30 * rate, axis=-2).reshape(history.shape).astype(history.dtype)
    return smooth, history - smooth
