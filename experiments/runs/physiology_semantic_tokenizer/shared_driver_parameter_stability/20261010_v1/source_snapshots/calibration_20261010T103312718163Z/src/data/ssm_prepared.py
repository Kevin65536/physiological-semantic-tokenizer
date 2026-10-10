"""Array access for the versioned public SSM prepared-feature contract.

Callers validate the exact parent root and permitted identities before calling
this boundary. This is not a raw-dataset parser and never rebuilds missing data.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def read_prepared_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        eeg, hb = arrays['eeg_features'].copy(), arrays['hb'].copy()
    if (eeg.ndim != 3 or eeg.shape[1:] != (120, 30)
            or hb.shape != (len(eeg), 120, 2)):
        raise ValueError('expected parent EEG [N,120,30] and Hb [N,120,2]')
    eeg.flags.writeable = False
    hb.flags.writeable = False
    return eeg, hb


def prepared_target(ref, coordinate):
    eeg, hb = read_prepared_arrays(ref['array_path'])
    index = ref['array_index']
    return np.column_stack((eeg[index]@np.asarray(coordinate['pc'])*coordinate['eeg_factor'],
                            hb[index]*coordinate['hb_factor']))


def fit_feature_coordinate(chunks, *, eeg_training_sd_target=.025, common_hb_training_sd_target=.025):
    """Fit PCA and the common Hb scale from caller-declared training chunks.

    The covariance center determines the PCA only; target reference subtraction
    belongs to the window-local operator. HbO/HbR retain their relative scale.
    Each chunk is (EEG [...,30], Hb [...,2]) on exactly the same support.
    """
    sx=np.zeros(30);xx=np.zeros((30,30));sh=np.zeros(2);hh=np.zeros(2);count=0
    for eeg,hb in chunks:
        x=np.asarray(eeg,float).reshape(-1,30);v=np.asarray(hb,float).reshape(-1,2)
        if len(x)!=len(v) or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError('coordinate training requires paired finite features')
        sx+=x.sum(axis=0);xx+=x.T@x;sh+=v.sum(axis=0);hh+=(v*v).sum(axis=0);count+=len(x)
    if not count:raise ValueError('empty coordinate training partition')
    covariance=xx/count-np.outer(sx/count,sx/count)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance);pc=eigenvectors[:,-1]
    if pc[np.argmax(abs(pc))]<0:pc=-pc
    variance_h=np.maximum(hh/count-(sh/count)**2,0.)
    if eigenvalues[-1]<=1e-16 or min(variance_h)<=1e-24:raise ValueError('degenerate training coordinate')
    eeg_factor=eeg_training_sd_target/np.sqrt(eigenvalues[-1])
    hb_factor=common_hb_training_sd_target/np.sqrt(np.mean(variance_h))
    return dict(pc=pc,eeg_factor=eeg_factor,hb_factor=hb_factor,
        sd=np.r_[np.sqrt(eigenvalues[-1])*eeg_factor,np.sqrt(variance_h)*hb_factor],
        training_samples=count,
        centering='first_5s_reference_only; covariance_center_used_for_PCA_not_subtracted_from_target',
        gain_interpretation='effective_in_frozen_dataset_site_gauge_not_absolute_physiology')
