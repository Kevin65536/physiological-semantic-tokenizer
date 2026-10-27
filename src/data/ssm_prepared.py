"""Array access for the versioned public SSM prepared-feature contract.

Callers validate the exact parent root and permitted identities before calling
this boundary. This is not a raw-dataset parser and never rebuilds missing data.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

EEG_PROXY_INPUT_SCHEMA = 'eeg_log_bandpower_10hz_v1'


def require_10hz_eeg_proxy_array(arrays, *, steps):
    """Reject old/upcast prepared targets before a current Tak fit."""
    schema = np.asarray(arrays.get('eeg_proxy_input', ''))
    rate = np.asarray(arrays.get('eeg_proxy_rate_hz', np.nan))
    clock = np.asarray(arrays.get('time_s', []), dtype=float)
    if (schema.shape != () or str(schema) != EEG_PROXY_INPUT_SCHEMA
            or rate.shape != () or float(rate) != 10.
            or np.shape(arrays.get('target')) != (steps, 3)
            or clock.shape != (steps,)
            or not np.allclose(clock, np.arange(steps)/10., rtol=0, atol=1e-12)):
        raise ValueError('Expected native-recomputed 10 Hz EEG proxy and matching clock; regenerate prepared arrays and training coordinates')


def require_cached_hb_array(arrays):
    """Do not join retained native targets to a new cached-input run."""
    from .unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    value = np.asarray(arrays.get('fnirs_input', ''))
    if value.shape != () or str(value) != CACHED_FNIRS_INPUT_SCHEMA:
        raise ValueError('Prepared Hb input is not unified_cached_fnirs_v1; regenerate targets and training coordinates in a fresh run')


@lru_cache(maxsize=16)
def public_prepared_datasets(parent):
    """REFED needs its resolved planar parent contract before any prepared read."""
    import yaml
    path=Path(parent)/'resolved_config.yaml'
    if path.is_file():
        cfg=yaml.safe_load(path.read_text())
        if cfg.get('schema')=='shared_driver_physiology_semantics_refed_v2':
            data=cfg['data']
            if (data.get('datasets')!=['refed'] or data.get('protected_data')!='forbidden'
                    or data.get('refed_geometry')!='refed_paper_layout_2d_v1' or not data.get('geometry_root')):
                raise ValueError('invalid REFED prepared parent data/geometry contract')
            return ('refed',)
    return ('eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation')


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


def pc1_band_contributions(features, coordinate):
    """Five signed contributions in the original six-channel PCA gauge.

    Features use channel-major/band-minor order and already carry the window
    reference subtraction. Summing these columns recovers the original PC1;
    neither a new center nor a new output SD is fitted here.
    """
    x = np.asarray(features, float)
    if x.ndim >= 2 and x.shape[-2:] == (6, 5):
        pass
    elif x.ndim >= 1 and x.shape[-1] == 30:
        x = x.reshape(*x.shape[:-1], 6, 5)
    else:
        raise ValueError('PC1 contributions require six channels by five bands')
    pc = np.asarray(coordinate['pc'], float)
    factor = float(coordinate['eeg_factor'])
    if (pc.shape != (30,) or not np.isfinite(pc).all()
            or not np.isfinite(x).all() or not np.isfinite(factor) or factor <= 0):
        raise ValueError('finite features and the frozen positive PCA scale required')
    return np.sum(x * pc.reshape(6, 5), axis=-2) * factor


def spatial_band_features(features, indices, *, eeg_factor=1.):
    """Mean channel log-power per band, preserving one frozen scale anchor."""
    x = np.asarray(features, float)
    take = np.asarray(indices)
    factor = float(eeg_factor)
    if (x.ndim < 2 or x.shape[-1] != 5 or not np.isfinite(x).all()
            or take.ndim != 1 or not len(take) or take.dtype.kind not in 'iu'
            or len(np.unique(take)) != len(take)
            or np.any(take < 0) or np.any(take >= x.shape[-2])
            or not np.isfinite(factor) or factor <= 0):
        raise ValueError('finite five-band features and unique channel indices required')
    return np.mean(x[..., take, :], axis=-2) * factor


def ring_band_features(features, inner_indices, outer_indices, *, alpha, eeg_factor=1.):
    """One common ring weight; endpoints retain the six/twelve-channel means."""
    inner, outer = np.asarray(inner_indices), np.asarray(outer_indices)
    if (inner.shape != (6,) or outer.shape != (6,)
            or np.intersect1d(inner, outer).size
            or not np.isfinite(alpha) or not 0 <= alpha <= .5):
        raise ValueError('disjoint six-channel rings and alpha in [0,.5] required')
    a = spatial_band_features(features, inner, eeg_factor=eeg_factor)
    b = spatial_band_features(features, outer, eeg_factor=eeg_factor)
    return (1 - alpha) * a + alpha * b


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
