"""Record-anchored orthogonal Fourier projections for 200 Hz EEG voltage."""
from functools import lru_cache

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


def continuous_indices(length, order):
    """DCT-II frequencies on the valid parent span, budgeted per elapsed time."""
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("Expected a positive sample count")
    return _continuous_indices(int(length), tuple(order))


@lru_cache(maxsize=256)
def _continuous_indices(length, order):
    if len(order) != 88 or set(order) != set(range(2, 90)):
        raise ValueError("Expected the frozen ranking of all 1–44.5 Hz cells")
    f = np.arange(length) * (RATE / 2 / length)
    budget = length * 64 // BLOCK
    core = (f >= 3.75) & (f <= 13.25)
    # Round to the nearest 0.5 Hz cell, ties to its upper neighbour.
    cells = np.floor(2*f + .5).astype(int)
    priority = np.full(length, len(order)+2, int)
    for rank, cell in enumerate(order):
        priority[cells == cell] = rank+1
    priority[core] = 0
    eligible = np.flatnonzero((f >= 1) & (f < 45) & (priority <= len(order)))
    pick = np.lexsort((f[eligible], abs(f[eligible]-cells[eligible]/2), priority[eligible]))
    keep = np.sort(eligible[pick[:budget]])
    keep.flags.writeable = False
    return keep


def continuous_projection(x, valid, order):
    """Independent real projection per continuous electrode support; no padding across gaps."""
    from scipy.fft import dct, idct
    x,valid=np.asarray(x),np.asarray(valid)
    if x.ndim!=2 or valid.shape!=x.shape or valid.dtype!=bool or not np.isfinite(x[valid]).all():
        raise ValueError('Expected finite valid time-by-channel voltages and a boolean mask')
    out=np.zeros(x.shape,dtype=np.result_type(x.dtype,np.float32))
    groups={}
    for p in range(x.shape[1]):
        edges=np.flatnonzero(np.diff(np.r_[False,valid[:,p],False]))
        for a,b in edges.reshape(-1,2):
            groups.setdefault((a,b),[]).append(p)
    for (a,b),sites in groups.items():
        keep=continuous_indices(b-a,order)
        c=dct(x[a:b,sites],axis=0,norm='ortho')
        drop=np.ones(len(c),bool);drop[keep]=False;c[drop]=0
        out[a:b,sites]=idct(c,axis=0,norm='ortho')
    return out
