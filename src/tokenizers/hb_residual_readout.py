"""Small frozen linear HbO residual readout from existing integer reconstructions."""
from pathlib import Path
import hashlib

import numpy as np
import torch

from .contextual_latent_dictionary import load_model as load_source


def features(prediction, name, lag=5):
    s, o, total = [np.asarray(prediction[key], float).reshape(len(prediction[key]),300,2)
                   for key in ('S','O','sum')]
    if name == 'x_gain':
        return total[:,:,:1]
    own = np.stack((s[:,:,0],o[:,:,0]),axis=-1)
    if name == 'same_color_so':
        return own
    pair = np.concatenate((own,np.stack((s[:,:,1],o[:,:,1]),axis=-1)),axis=-1)
    if name == 'paired_so':
        return pair
    if name != 'paired_so_context' or not 0 < lag < 300:
        raise ValueError('known fixed Hb residual feature layout required')
    padded = np.pad(pair,((0,0),(lag,lag),(0,0)),mode='edge')
    context = .5*(padded[:,:300]+padded[:,2*lag:2*lag+300])-pair
    return np.concatenate((pair,context),axis=-1)


def apply_readout(prediction, quiet, datasets, fitted):
    result = {key:np.asarray(prediction[key]).copy() for key in ('S','O','sum')}
    f = features(prediction,fitted['feature_set'],fitted['context_lag_samples'])
    f = np.where(np.asarray(quiet)[:,None,None],0.,f)
    residual = np.zeros((len(f),300))
    group = np.full(len(f),'*',dtype=object) if fitted['grouping']=='global' else np.asarray(datasets)
    for key in sorted(set(group)):
        if key not in fitted['coefficients']:
            raise ValueError('dataset identity is outside the frozen readout')
        coefficient = np.asarray(fitted['coefficients'][key],float)
        if coefficient.shape != (f.shape[-1],) or not np.isfinite(coefficient).all():
            raise ValueError('finite fixed residual coefficients required')
        selected = group == key
        residual[selected] = f[selected] @ coefficient
    other = result['O'].reshape(len(f),300,2).astype(float)
    total = result['sum'].reshape(len(f),300,2).astype(float)
    other[:,:,0] += residual
    total[:,:,0] += residual
    result['O'] = other.reshape(result['O'].shape)
    result['sum'] = total.reshape(result['sum'].shape)
    np.testing.assert_allclose(result['S']+result['O'],result['sum'],rtol=1e-12,atol=1e-12)
    return result


class HbResidualReadout:
    def __init__(self, source, metadata, fitted):
        self.source, self.metadata, self.fitted = source, metadata, fitted
        if not hasattr(source,'silent_ids'):
            raise ValueError('frozen reserved-pair Hb source required')

    def encode(self, values):
        return self.source.encode(values)

    def decode_ids(self, ids, datasets=None):
        prediction = self.source.decode_ids(ids)
        n = len(ids['S'])
        if datasets is None:
            if self.fitted['grouping'] != 'global':
                raise ValueError('fixed dataset identities required by this readout')
            datasets = ['*']*n
        if len(datasets) != n:
            raise ValueError('one fixed dataset identity per integer sequence required')
        quiet = np.ones(n,dtype=bool)
        for branch in ('S','O'):
            quiet &= (np.asarray(ids[branch]) == self.source.silent_ids[branch]).all(1)
        result = apply_readout(prediction,quiet,datasets,self.fitted)
        for branch in ('S','O','sum'):
            np.testing.assert_array_equal(result[branch][quiet],prediction[branch][quiet])
        return result


def load_model(path):
    path = Path(path)
    saved = torch.load(path,map_location='cpu',weights_only=False)
    metadata = saved['metadata']
    if metadata['schema'] != 'hb_x_residual_v1':
        raise ValueError('frozen Hb residual readout schema required')
    source_path = (path.parent/metadata['source_file']).resolve()
    if source_path.parent != path.parent.resolve() or hashlib.sha256(source_path.read_bytes()).hexdigest() != metadata['source_sha256']:
        raise ValueError('accompanying frozen Hb source changed')
    return HbResidualReadout(load_source(source_path),metadata,saved['fitted'])
