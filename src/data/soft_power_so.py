"""Smooth bounded EEG carrier gains with a soft power penalty on final voltage."""
from functools import lru_cache

import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize
from scipy.signal import butter, sosfreqz

from .dual_component_tokenizer import BANDS

RATE = 200
HOP = 20


@lru_cache(maxsize=8)
def responses(n):
    # The orthonormal cosine basis implements a symmetric even extension.
    freq = np.arange(n) * RATE / (2 * n)
    weights = []
    for band in BANDS:
        sos = butter(4, band, fs=RATE, btype="bandpass", output="sos")
        _, h = sosfreqz(sos, worN=freq, fs=RATE)
        weights.append(abs(h) ** 2)
    return np.array([*weights, 1 - np.sum(weights, axis=0)])


def frame_power(filtered):
    """One-second, uncentered power at 0.5..duration-0.5 s, inclusive."""
    sums = (filtered ** 2).reshape(6, -1, HOP).sum(axis=2)
    acc = np.pad(np.cumsum(sums, axis=1), ((0, 0), (1, 0)))
    return (acc[:, 10:] - acc[:, :-10]) / RATE


def final_power(voltage):
    return frame_power(idct(responses(len(voltage)) * dct(voltage, norm="ortho"),
                            norm="ortho"))


class GainObjective:
    """One electrode, with both voltage and reference in train-RMS units."""
    def __init__(self, carriers, ref, delta, knot_seconds, weight):
        self.carriers = np.ascontiguousarray(carriers.T)
        self.x = carriers.sum(axis=1)
        n = len(self.x)
        stride = round(RATE * knot_seconds)
        if (n % HOP or stride < HOP or carriers.shape != (n, 6)
                or ref.shape != (n // HOP - 9, 6) or not np.all(ref > 0)
                or not np.isfinite(ref).all() or not np.isfinite(carriers).all()
                or delta < 0 or weight <= 0):
            raise ValueError("Invalid EEG voltage, frame reference or gain basis")
        self.knots = np.r_[np.arange(0, n - 1, stride), n - 1]
        self.left = np.searchsorted(self.knots[1:], np.arange(n), side="right").clip(max=len(self.knots)-2)
        self.frac = (np.arange(n) - self.knots[self.left]) / np.diff(self.knots)[self.left]
        self.threshold = np.log(ref.T + 1e-12) + delta
        self.weight = weight

    def voltage(self, params):
        g = params.reshape(6, -1)
        return ((g[:, self.left] * (1 - self.frac) + g[:, self.left + 1] * self.frac)
                * self.carriers).sum(axis=0)

    def evaluate(self, params):
        s = self.voltage(params)
        transfer = responses(len(s))
        filtered = idct(transfer * dct(s, norm="ortho"), norm="ortho")
        power = frame_power(filtered).clip(min=0) + 1e-12
        excess = np.maximum(np.log(power) - self.threshold, 0)
        err = s - self.x
        loss = np.mean(err ** 2) + self.weight * np.mean(excess ** 2)
        adj = (2 * self.weight / (excess.size * RATE)) * excess / power
        # Transpose of the ten-bin moving sum: each frame contributes to ten adjacent bins.
        summed = np.cumsum(np.pad(adj, ((0, 0), (1, 10))), axis=1)
        bin_adj = summed[:, 1:] - np.pad(summed[:, :-10], ((0, 0), (10, 0)))[:, 1:]
        grad_f = 2 * filtered * np.repeat(bin_adj[:, :-1], HOP, axis=1)
        grad_s = 2 * err / len(s) + idct((transfer * dct(grad_f, norm="ortho")).sum(axis=0), norm="ortho")
        dg = self.carriers * grad_s
        grad = np.array([np.bincount(self.left, row * (1-self.frac), minlength=len(self.knots))
                         + np.bincount(self.left+1, row * self.frac, minlength=len(self.knots)) for row in dg])
        return loss, grad.ravel()


def fit_soft_channel(carriers, reference, train_rms, delta, knot_seconds=5., weight=1.):
    if not np.isfinite(train_rms) or train_rms <= 0:
        raise ValueError("Positive frozen training RMS required")
    objective = GainObjective(carriers / train_rms, reference / train_rms**2, delta, knot_seconds, weight)
    start = np.full(6 * len(objective.knots), .7)
    initial, _ = objective.evaluate(start)
    fit = minimize(objective.evaluate, start, jac=True, method="L-BFGS-B",
                   bounds=[(0., 1.)] * len(start),
                   options=dict(maxiter=100, ftol=1e-7, gtol=1e-5))
    if not np.isfinite(fit.fun) or fit.fun > initial + 1e-12 or fit.status not in (0, 1):
        raise RuntimeError(f"Soft power fit failed: {fit.message}")
    return objective.voltage(fit.x) * train_rms, fit.x.reshape(6, -1), dict(
        initial=float(initial), final=float(fit.fun), iterations=fit.nit, converged=fit.success)
