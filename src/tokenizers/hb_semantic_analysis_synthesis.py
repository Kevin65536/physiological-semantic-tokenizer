"""Observation-only O-code search inside a training-defined semantic neighborhood."""
from pathlib import Path
import hashlib

import numpy as np
import torch


class HbSemanticAnalysisSynthesis:
    def __init__(self,source,metadata,neighbors,x_variance,shape_variance,slope_variance,o_change_variance=None):
        self.source = source
        self.metadata = metadata
        self.codebook_size = int(metadata['codebook_size'])
        self.neighbors = np.asarray(neighbors,dtype=np.int64)
        self.x_variance = np.asarray(x_variance,dtype=float)
        self.shape_variance = np.asarray(shape_variance,dtype=float)
        self.slope_variance = np.asarray(slope_variance,dtype=float)
        self.change_penalty = float(metadata.get('search_change_penalty',0.))
        self.o_change_variance = np.asarray(x_variance if o_change_variance is None else o_change_variance,dtype=float)
        self.clock = (np.arange(20)-9.5)/10
        size = int(metadata['branch_codebook_sizes']['O'])
        if (self.neighbors.shape[0] != size or self.neighbors.ndim != 2
                or (self.neighbors<0).any() or (self.neighbors>=size).any()
                or not np.array_equal(self.neighbors[:,0],np.arange(size))
                or any(value.shape != (2,) or not np.isfinite(value).all() or (value<=0).any()
                       for value in (self.x_variance,self.shape_variance,self.slope_variance,self.o_change_variance))
                or not np.isfinite(self.change_penalty) or self.change_penalty<0):
            raise ValueError('finite training scales and in-range anchored semantic neighborhoods required')
        if source.metadata.get('decoder_color_gain') is not None or source.metadata.get('decoder_lowpass_hz') is not None:
            raise ValueError('raw frozen prototype decoder required for the matched O search')

    def error(self,predicted,observed):
        difference = np.asarray(predicted,float).reshape(-1,300,2)-np.asarray(observed,float).reshape(-1,300,2)
        return np.mean(difference*difference/self.x_variance,axis=(1,2))

    def semantic_drift(self,predicted,baseline):
        p = np.asarray(predicted,float).reshape(-1,15,20,2)
        b = np.asarray(baseline,float).reshape(-1,15,20,2)
        p = p-p.mean(2,keepdims=True)
        b = b-b.mean(2,keepdims=True)
        difference = p-b
        shape = np.mean(difference*difference,axis=(1,2))/self.shape_variance
        slope = np.einsum('bptc,t->bpc',difference,self.clock)/np.sum(self.clock**2)
        slope = np.mean(slope*slope,axis=1)/self.slope_variance
        return np.c_[shape,slope]

    def decode_o(self,ids):
        model = self.source.model
        quantizer = model.branches['O']['quantizer']
        with torch.inference_mode():
            indices = torch.as_tensor(ids,dtype=torch.long)
            return model._decode_component('O',quantizer.get_embedding(indices)).numpy()

    def encode(self,values):
        observed = np.asarray(values)
        original = self.source.encode(observed)
        baseline = self.source.decode_ids(original)
        selected = original['O'].copy()
        width = min(int(self.metadata['neighbor_budget']),self.neighbors.shape[1])
        candidates = self.neighbors[original['O'],:width]
        limit = float(self.metadata['maximum_normalized_semantic_drift'])
        for _ in range(int(self.metadata['coordinate_passes'])):
            for patch in range(15):
                trial = np.repeat(selected[:,None,:],width,axis=1)
                trial[:,:,patch] = candidates[:,patch]
                prediction = self.decode_o(trial.reshape(-1,15))
                repeated_o = np.repeat(baseline['O'],width,axis=0)
                drift = self.semantic_drift(prediction,repeated_o)
                allowed = (drift<=limit).all(axis=1).reshape(len(observed),width)
                s = np.repeat(baseline['S'],width,axis=0)
                x = np.repeat(observed,width,axis=0)
                errors = self.error(s+prediction,x).reshape(len(observed),width)
                if self.change_penalty:
                    change = np.asarray(prediction,float).reshape(-1,300,2)-np.asarray(repeated_o,float).reshape(-1,300,2)
                    errors += self.change_penalty*np.mean(change*change/self.o_change_variance,axis=(1,2)).reshape(len(observed),width)
                errors = np.where(allowed,errors,np.inf)
                best = errors.argmin(axis=1)
                selected = trial[np.arange(len(observed)),best]
        result = dict(S=original['S'],O=selected)
        decoded = self.source.decode_ids(result)
        allowed = (self.semantic_drift(decoded['O'],baseline['O'])<=limit).all(axis=1)
        allowed &= self.error(decoded['sum'],observed)<=self.error(baseline['sum'],observed)
        result['O'][~allowed] = original['O'][~allowed]
        return result

    def decode_ids(self,ids):
        return self.source.decode_ids(ids)

    def predict_continuous(self,values):
        return self.source.predict_continuous(values)

    def __call__(self,values):
        ids = self.encode(values)
        return dict(self.decode_ids(ids),ids=ids)


def load_model(path):
    from .contextual_latent_dictionary import load_model as load_source
    path = Path(path)
    saved = torch.load(path,map_location='cpu',weights_only=False)
    metadata = saved['metadata']
    if metadata['schema'] != 'hb_semantic_analysis_synthesis_v1':
        raise ValueError('Hb semantic analysis-synthesis schema required')
    source_path = (path.parent/metadata['source_file']).resolve()
    if source_path.parent != path.parent.resolve() or hashlib.sha256(source_path.read_bytes()).hexdigest() != metadata['source_sha256']:
        raise ValueError('accompanying frozen source identity changed')
    return HbSemanticAnalysisSynthesis(load_source(source_path),metadata,saved['neighbors'],saved['x_variance'],
        saved['shape_variance'],saved['slope_variance'],saved.get('o_change_variance'))
