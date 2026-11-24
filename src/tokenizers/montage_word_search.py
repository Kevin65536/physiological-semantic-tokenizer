"""Select existing paid words by their actual joint effect on a frozen native EEG montage."""
import numpy as np

from .cross_branch_innovations import PILOT_SIZE, condition_on_pilots, pilot_positions


def project(waves):
    shape = waves.shape
    spectrum = np.fft.rfft(waves.reshape(shape[0], -1))
    frequency = np.fft.rfftfreq(shape[1] * shape[2], 1 / 200)
    spectrum[:, (frequency < 1) | (frequency > 45)] = 0
    return np.fft.irfft(spectrum, n=shape[1] * shape[2]).reshape(shape)


def synthesis_matrix(roles, state):
    """Columns multiply an energy-scaled pilot atom or an unscaled residual atom."""
    n = len(roles) // 2
    pilots = pilot_positions(n)
    mapping, conditional = condition_on_pilots(state['joint'], roles, state['pilot_gain'], state['pilot_noise'])
    matrix = np.zeros((2*n, 2*n))
    for band, (sites, operator) in enumerate(mapping):
        matrix[:, sites] = operator
        second = np.flatnonzero((roles == band) & ~pilots)
        gain = state['innovation_mean_gain'][band, second]
        cov = conditional[band]
        measured = gain[:, None] * cov[np.ix_(second, second)] * gain + np.diag(state['innovation_mean_noise'][band, second])
        active = np.diag(measured) > 1e-20
        if active.any():
            sites = second[active]
            coeff = np.linalg.solve(measured[np.ix_(active,active)], gain[active,None]*cov[sites]).T
            matrix[:, sites] = coeff * state['scale'][band,sites]
    return np.vstack((matrix, matrix[:n] + matrix[n:]))


def dictionaries(pilot_books, second_books):
    output = {}
    for b in 'SO':
        for tag in range(5):
            for pilot, books, mode in [(True,pilot_books,'balanced'), (False,second_books,'innovation')]:
                book = books[b][tag]
                atoms = np.asarray(book[mode+'_atoms'], float)
                if pilot:
                    atoms = atoms * book['balanced_gain'][:,None]
                output[b,tag,pilot] = dict(atoms=atoms, fast=np.ascontiguousarray(atoms.T, dtype=np.float32),
                    norm=np.einsum('ij,ij->i',atoms,atoms), property_centers=book[mode+'_centers'][:,-3:],
                    mean=book['property_mean'], scale=book['property_scale'], count=book[mode+'_count'],
                    offset=1+tag*1023 if pilot else PILOT_SIZE+tag*int(book['capacity']))
    return output


def local_spectrum(waves):
    """A tight, zero-extended 1 s sine-window analysis on 0.5 s hops."""
    signal=waves.reshape(len(waves),-1)
    padded=np.pad(signal,((0,0),(100,100)))
    frames=np.lib.stride_tricks.sliding_window_view(padded,200,axis=1)[:,::100]
    window=np.sin(np.pi*(np.arange(200)+.5)/200)
    return np.fft.rfft(frames*window,axis=-1,norm='ortho')


def spectrum_adjoint(spectrum, shape):
    frames=np.fft.irfft(spectrum,n=200,axis=-1,norm='ortho')
    frames*=np.sin(np.pi*(np.arange(200)+.5)/200)
    result=np.zeros((shape[0],shape[1]*shape[2]+200))
    for i in range(frames.shape[1]):result[:,100*i:100*i+200]+=frames[:,i]
    return result[:,100:-100].reshape(shape)


def spectral_residual(target, prediction, target_magnitude, strength):
    """A majorizing quadratic for time-domain error plus local magnitude error."""
    error=target-prediction
    if not strength:return project(error), np.square(error).sum((1,2))
    spectrum=local_spectrum(prediction);magnitude=abs(spectrum)
    hz=np.fft.rfftfreq(200,1/200);metric=np.where((hz>=4)&(hz<13),2.,1.)*strength
    phase=np.divide(spectrum,magnitude,out=np.ones_like(spectrum),where=magnitude>0)
    delta=target_magnitude-magnitude
    direction=error+spectrum_adjoint(delta*phase*metric,prediction.shape)
    # rFFT retains only the nonnegative half of the spectrum of real samples.
    multiplicity=np.r_[1.,np.full(99,2.),1.]
    loss=np.square(error).sum((1,2))+(np.square(delta)*metric*multiplicity).sum((1,2))
    return project(direction)/(1+metric.max()),loss


def select_words(sent, matrix, target, weight, properties, table, roles, limit, sweeps, spectral_strength=0.):
    """Each sweep decreases a quadratic upper bound touching the exact band-limited loss."""
    n = len(sent['S'])
    paid = np.concatenate([sent[b] for b in 'SO']).astype(np.int32)
    pilots = pilot_positions(n)
    active = np.flatnonzero(roles >= 0)
    choices = {}
    values = np.zeros((2*n,15,400))
    gram = matrix.T @ (weight[:,None]*matrix)
    for j in active:
        b = 'SO'[j//n]; book = table[b,int(roles[j]),bool(pilots[j])]
        current = paid[j]-book['offset']
        values[j] = book['atoms'][current]
        if limit:
            prop = (properties[b][j%n]-book['mean'])/(np.sqrt(3)*book['scale'])
            dist = ((prop[:,None,:]-book['property_centers'][None,:,:])**2).sum(-1)
            dist[:,book['count']==0] = np.inf
            count=min(limit,int(np.sum(book['count']>0)))
            choices[j] = np.argpartition(dist,count-1,axis=1)[:,:count]
    def decode():
        return (matrix @ values.reshape(2*n,6000)).reshape(-1,15,400)
    before = project(decode())
    target_magnitude=abs(local_spectrum(target)) if spectral_strength else None
    score_residual, per_channel=spectral_residual(target,before,target_magnitude,spectral_strength)
    losses=[float(weight@per_channel)]
    changed = []
    for step in range(sweeps):
        # P is symmetric and idempotent. P(target - P(current)) is the score residual.
        driving = (matrix.T @ (weight[:,None]*score_residual.reshape(3*n,-1))).reshape(2*n,15,400)
        movement = np.zeros_like(values); updates = 0
        for j in (active if step%2==0 else active[::-1]):
            if gram[j,j]<=0:continue
            book = table['SO'[j//n],int(roles[j]),bool(pilots[j])]
            current = paid[j]-book['offset']
            local = values[j]+(driving[j]-(gram[j]@movement.reshape(2*n,-1)).reshape(15,400))/gram[j,j]
            scores = book['norm']-2*(local.astype(np.float32)@book['fast'])
            if limit:
                subset = choices[j]
                pick = subset[np.arange(15),np.argmin(np.take_along_axis(scores,subset,axis=1),axis=1)]
            else:
                scores[:,book['count']==0] = np.inf
                pick = scores.argmin(1)
            selected = book['atoms'][pick]
            keep = np.sum((selected-local)**2,1) < np.sum((values[j]-local)**2,1)
            pick = np.where(keep,pick,current)
            next_values = book['atoms'][pick]
            movement[j] = next_values-values[j]
            updates += np.count_nonzero(pick!=current)
            values[j] = next_values
            paid[j] = pick+book['offset']
        before = project(decode())
        score_residual, per_channel=spectral_residual(target,before,target_magnitude,spectral_strength)
        loss=float(weight@per_channel)
        if loss>losses[-1]*(1+1e-10)+1e-10:
            raise ArithmeticError('joint word update increased the visible-source objective')
        losses.append(loss);changed.append(int(updates))
        if not updates:break
    return dict(S=paid[:n].astype(np.uint16),O=paid[n:].astype(np.uint16)),dict(loss=losses,updates=changed)
