"""Record-anchored orthogonal Fourier projections for 200 Hz EEG voltage."""
import numpy as np

from .dual_component_tokenizer import BANDS

BLOCK = 400
RATE = 200


def complete_blocks(x, origin):
    """Only full absolute-grid blocks have a target; never synthesize missing context."""
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all() or origin < 0 or int(origin) != origin:
        raise ValueError("Finite time-by-electrode EEG and original integer origin required")
    first = int(-origin % BLOCK)
    count = (len(x)-first)//BLOCK
    if count < 1:
        raise ValueError("No complete 2 s block")
    section = slice(first, first+BLOCK*count)
    return x[section].reshape(count, BLOCK, x.shape[1]), section


def project_blocks(blocks, bins):
    bins = np.asarray(bins)
    if (blocks.ndim != 3 or blocks.shape[1] != BLOCK or bins.ndim != 1
            or bins.dtype.kind not in 'iu' or not len(bins) or np.any((bins < 1) | (bins >= BLOCK//2))
            or len(np.unique(bins)) != len(bins)):
        raise ValueError("Expected complete blocks and unique positive interior Fourier bins")
    coefficients = np.fft.rfft(blocks, axis=1)
    keep = np.zeros(coefficients.shape[1], bool)
    keep[bins] = True
    coefficients[:, ~keep] = 0
    return np.fft.irfft(coefficients, n=BLOCK, axis=1).reshape(-1, blocks.shape[-1])


def rank_frequencies(mean_spectrum):
    """Nested phase-complete real subspaces with a fixed band allocation order."""
    freq = np.fft.rfftfreq(BLOCK, 1/RATE)
    groups = []
    for lo, hi in BANDS:
        indices = np.flatnonzero((freq >= lo) & (freq < hi))
        groups.append(sorted(indices, key=lambda k: (-mean_spectrum[k], k)))
    order = []
    while any(groups):
        for band in (0, 1, 2, 3, 1, 2, 3, 4):
            if groups[band]:
                order.append(int(groups[band].pop(0)))
    return order
