"""Frozen word profiles, short-sequence controls and paired development summaries."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from numba import njit


def wave_attribute(wave, mod):
    if mod == 'hb':
        t = (np.arange(20)-9.5)/10
        return wave.reshape(*wave.shape[:-1],20,2)[...,0] @ (t/(t@t))
    centered = wave - wave.mean(-1,keepdims=True)
    power = np.abs(np.fft.rfft(centered*np.hanning(400),axis=-1))**2
    total = power[...,2:91].sum(-1)
    return np.divide(power[...,16:26].sum(-1), total, out=np.zeros_like(total), where=total>0)


def profile(ids, y, size):
    count = np.bincount(ids,minlength=size)
    score = (np.bincount(ids,weights=y,minlength=size)+16*y.mean())/(count+16)
    # Fit empirical boundaries on emitted tokens, with stable ties across splits.
    bounds = np.quantile(score[ids],[.25,.5,.75])
    return np.searchsorted(bounds,score,side='right'), count, score, bounds


def descriptors(seqs):
    n, length = seqs.shape
    def hist(code, width):
        bins = code + np.arange(n)[:,None]*width
        return np.bincount(bins.ravel(),minlength=n*width).reshape(n,width)/code.shape[1]
    bag = hist(seqs,4)
    pairs = hist(seqs[:,:-1]*4+seqs[:,1:],16)
    triples = hist(seqs[:,:-2]*16+seqs[:,1:-1]*4+seqs[:,2:],64)
    # Counts of runs of length 1, 2, >=3, per word, normalized by sequence length.
    row, col = np.nonzero(np.column_stack([np.ones(n,bool),seqs[:,1:]!=seqs[:,:-1]]))
    lengths = np.diff(np.r_[row*length+col,n*length])
    dwell = np.bincount(row*12+seqs[row,col]*3+np.minimum(lengths,3)-1,minlength=n*12).reshape(n,12)/length
    return bag,np.column_stack((bag,pairs,dwell)),np.column_stack((bag,pairs,dwell,triples))


@njit
def surrogate_batch(seqs, mode, seed):
    """0 = bag, 1 = exact directed Euler trail, 2 = shuffle lengths within word."""
    np.random.seed(seed)
    output = seqs.copy()
    for i in range(len(seqs)):
        x, y = seqs[i], output[i]
        n = len(x)
        if mode == 0:
            np.random.shuffle(y)
        elif mode == 1:
            edges = np.empty((4,n),np.int64)
            count = np.zeros(4,np.int64)
            for t in range(n-1):
                edges[x[t],count[x[t]]] = x[t+1]
                count[x[t]] += 1
            for w in range(4):
                np.random.shuffle(edges[w,:count[w]])
            stack = np.empty(n,np.int64)
            stack[0] = x[0]
            depth, end = 1,n-1
            while depth:
                w = stack[depth-1]
                if count[w]:
                    count[w] -= 1
                    stack[depth] = edges[w,count[w]]
                    depth += 1
                else:
                    y[end] = w
                    depth -= 1
                    end -= 1
        elif mode == 2:
            starts = np.concatenate((np.array([0]), np.flatnonzero(x[1:]!=x[:-1])+1))
            words = x[starts]
            lengths = np.diff(np.concatenate((starts,np.array([n]))))
            for w in range(4):
                idx = np.flatnonzero(words==w)
                shuffled = lengths[idx]
                np.random.shuffle(shuffled)
                lengths[idx] = shuffled
            offset = 0
            for j in range(len(starts)):
                y[offset:offset+lengths[j]] = words[j]
                offset += lengths[j]
        else:
            raise ValueError('unknown surrogate')
    return output


def fit_ridge(x, y):
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=.01*len(x)).fit(scaler.transform(x),y)
    return lambda values: model.predict(scaler.transform(values))


def interval(values, datasets, seed=712):
    values, datasets = np.asarray(values), np.asarray(datasets)
    if not len(values):
        return [None,None,None]
    rng = np.random.default_rng(seed)
    groups = [values[datasets==d] for d in np.unique(datasets)]
    draws = np.mean([rng.choice(g,(2000,len(g))).mean(1) for g in groups],axis=0)
    return [float(np.mean([g.mean() for g in groups])), *np.quantile(draws,[.025,.975]).tolist()]
