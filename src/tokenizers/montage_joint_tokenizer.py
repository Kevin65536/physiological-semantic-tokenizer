"""Joint hard-atomic EEG assignment and full-montage integer-only synthesis."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, spectral_features


class MontageNetwork(nn.Module):
    def __init__(self, source, metadata):
        super().__init__()
        self.metadata = metadata
        self.atoms = nn.ParameterDict()
        self.centers = nn.ParameterDict()
        self.bias = nn.ParameterDict()
        self.attributes = nn.ParameterDict()
        self.synthesis = nn.ParameterDict()
        self.analysis = nn.ParameterDict()
        self.temperature = metadata.get('temperature', .05)
        for branch in ('S', 'O'):
            for band in range(5):
                key = f'{branch}_{band}'
                self.atoms[key] = nn.Parameter(torch.tensor(source.atoms[branch, band], dtype=torch.float32))
                self.centers[key] = nn.Parameter(torch.tensor(source.centers[branch, band], dtype=torch.float32))
                self.bias[key] = nn.Parameter(torch.zeros(len(source.centers[branch, band])))
                self.attributes[key] = nn.Parameter(torch.zeros(len(source.atoms[branch, band]), 3))
                if metadata.get('global_analysis'):
                    self.register_buffer('retrieval_'+key,torch.tensor(source.centers[branch,band],dtype=torch.float32))
                    self.register_buffer('carrier_weight_'+key,torch.tensor(source.feature_weights[branch,band][:-3],dtype=torch.float32))
        for dataset, names in metadata['positions'].items():
            count = len(names)
            roles = np.arange(count) % 5
            for band in range(5):
                matrix = np.zeros((2*count, 2*count), np.float32)
                for b, branch in enumerate(('S', 'O')):
                    positions = np.flatnonzero(roles == band)
                    # Inputs are matrix rows; outputs are matrix columns.
                    local = np.zeros((count, count))
                    local[positions, :] = source.weights[dataset, branch, band]
                    matrix[b*count:(b+1)*count, b*count:(b+1)*count] = local
                self.synthesis[f'{dataset}_{band}'] = nn.Parameter(torch.from_numpy(matrix))
                if metadata.get('global_analysis'):
                    selected=np.flatnonzero(roles==band)
                    analysis=np.zeros((2*count,2*len(selected)),np.float32)
                    for b in range(2):
                        analysis[b*count+selected,b*len(selected)+np.arange(len(selected))]=1.
                    self.analysis[f'{dataset}_{band}']=nn.Parameter(torch.from_numpy(analysis))

    def assign(self, features, candidates, observed, dataset, *, straight_through=True, decode_atoms=True):
        count = len(self.metadata['positions'][dataset])
        roles = np.arange(count) % 5
        decoded = torch.zeros((2, count, 15, 400))
        attributes = torch.zeros((2, count, 15, 3))
        ids = {}
        global_query={}
        if self.metadata.get('global_analysis'):
            for band in range(5):
                carrier=features[f'global_{band}']
                mixed=self.analysis[f'{dataset}_{band}'].T @ carrier.reshape(2*count,-1)
                global_query[band]=mixed.reshape(2,-1,15,carrier.shape[-1])
        for b, branch in enumerate(('S', 'O')):
            ids[branch] = torch.zeros((count, 15), dtype=torch.long)
            for band in range(5):
                positions = np.flatnonzero(roles == band)
                key = f'{branch}_{band}'
                feature = features[key]
                candidate = candidates[key]
                if self.metadata.get('global_analysis'):
                    carrier=global_query[band][b].reshape(len(feature),-1)*getattr(self,'carrier_weight_'+key)
                    feature=torch.cat((carrier,feature[:,-3:]),dim=-1)
                    initial=getattr(self,'retrieval_'+key)
                    with torch.no_grad():
                        distance=feature.detach().square().sum(1,keepdim=True)+initial.square().sum(1)-2*feature.detach()@initial.T
                        candidate=torch.topk(distance,min(self.metadata.get('top_m',16),len(initial)),dim=1,largest=False,sorted=True).indices
                center = self.centers[key][candidate]
                logits = -torch.sum((feature[:, None]-center)**2, dim=-1)/self.temperature + self.bias[key][candidate]
                soft = torch.softmax(logits, dim=-1)
                chosen = logits.argmax(-1)
                hard = torch.nn.functional.one_hot(chosen, candidate.shape[-1]).to(soft.dtype)
                weights = hard + (soft-soft.detach()) if straight_through else hard
                available = observed[positions, None].expand(len(positions), 15).reshape(-1)
                weights = weights*available[:, None]
                selected = candidate.gather(1, chosen[:, None]).squeeze(1)+1
                ids[branch][positions] = (selected*available).reshape(len(positions), 15)
                if decode_atoms:
                    atom = self.atoms[key][candidate+1]
                    decoded[b, positions] = torch.einsum('nm,nmt->nt', weights, atom).reshape(len(positions), 15, 400)
                    attr = self.attributes[key][candidate+1]
                    attributes[b, positions] = torch.einsum('nm,nma->na', weights, attr).reshape(len(positions), 15, 3)
        return decoded, attributes, ids

    def decode(self, atoms, dataset, *, spatial=True, observed=None):
        count = atoms.shape[1]
        spectrum = torch.fft.rfft(atoms.reshape(2*count, 6000), dim=-1)
        result = torch.zeros_like(spectrum)
        for band, (lower, upper) in enumerate(BANDS):
            start, stop = int(lower*30), int(upper*30)
            values = spectrum[:, start:stop]
            if spatial:
                matrix = self.synthesis[f'{dataset}_{band}']
                if observed is not None and not bool(observed[np.arange(count) % 5 == band].any()):
                    matrix = torch.eye(2*count)
                values = matrix.T.to(values.dtype) @ values
            result[:, start:stop] = values
        return torch.fft.irfft(result, n=6000, dim=-1).reshape(2, count, 15, 400)

    def decode_ids(self, ids, dataset, *, spatial=True):
        count = len(self.metadata['positions'][dataset])
        roles = np.arange(count) % 5
        decoded = torch.zeros((2, count, 15, 400))
        for b, branch in enumerate(('S', 'O')):
            for band in range(5):
                positions = np.flatnonzero(roles == band)
                decoded[b, positions] = self.atoms[f'{branch}_{band}'][ids[branch][positions]]
        observed = (ids['S'] != 0).any(-1) | (ids['O'] != 0).any(-1)
        return self.decode(decoded, dataset, spatial=spatial, observed=observed)


class MontageJointTokenizer(SpatialWaveformDictionary):
    """Frozen retrieval scaffold plus the learned ``network`` parameter owner.

    Inherited arrays retain the initialization for exact candidate retrieval and
    self-contained loading. The learned atoms and complete synthesis matrices
    live in ``network``; every integer decode reads those learned parameters.
    """
    def __init__(self, source, metadata=None):
        metadata = deepcopy(metadata or source.metadata)
        metadata['schema'] = ('montage_joint_phase_readout_v1' if metadata.get('integer_phase_readout') else 'montage_joint_tokenizer_v1')
        metadata['decoder_continuous_side_information'] = 0
        super().__init__(metadata, source.transfer, source.centers, source.atoms, source.weights,
                         feature_weights=source.feature_weights)
        if not self.feature_weights:
            raise ValueError('whole-patch semantic and band-phase initialization required')
        self.network = MontageNetwork(source, metadata)
        self.phase_readout = {}

    def shortlist(self, prediction, branch, band):
        features = self.assignment_features(prediction, branch, band).astype(np.float32)
        centers = self.centers[branch, band].astype(np.float32)
        distance = np.square(features).sum(1, keepdims=True)+np.square(centers).sum(1)-2*features@centers.T
        m = min(self.metadata.get('top_m', 16), len(centers))
        candidates = np.argsort(distance, axis=1, kind='stable')[:, :m].astype(np.int64)
        return features, candidates

    def training_features(self, values, dataset, *, initial_candidates=True):
        prediction = self.predict_continuous(values)
        features, candidates = {}, {}
        if self.metadata.get('global_analysis'):
            for band in range(5):
                carrier=np.stack([spectral_features(prediction[b].reshape(-1,400),band).reshape(len(self.roles(dataset)),15,-1)
                                  for b in ('S','O')]).astype(np.float32)
                features[f'global_{band}']=torch.from_numpy(carrier)
        for branch in ('S', 'O'):
            for band in range(5):
                positions = np.flatnonzero(self.roles(dataset) == band)
                patches=prediction[branch][:,positions].reshape(-1,400)
                if self.metadata.get('global_analysis') and not initial_candidates:
                    f=self.assignment_features(patches,branch,band).astype(np.float32)
                    c=np.zeros((len(f),1),np.int64)
                else:
                    f,c=self.shortlist(patches,branch,band)
                features[f'{branch}_{band}'] = torch.from_numpy(f)
                candidates[f'{branch}_{band}'] = torch.from_numpy(c)
        return features, candidates

    def encode(self,values,dataset,observed):
        if not self.metadata.get('global_analysis'):
            return super().encode(values,dataset,observed)
        values=np.asarray(values);observed=np.asarray(observed)
        if values.shape[1:]!=(len(self.roles(dataset)),15,400) or observed.shape!=values.shape[:2] or observed.dtype!=bool:
            raise ValueError('complete fixed native montage and observation-only availability required')
        values=np.where(observed[:,:,None,None],values,0.)
        result={b:[] for b in ('S','O')}
        with torch.no_grad():
            for n in range(len(values)):
                features,candidates=self.training_features(values[n:n+1],dataset,initial_candidates=False)
                _,_,ids=self.network.assign(features,candidates,torch.from_numpy(observed[n]),dataset,straight_through=False,decode_atoms=False)
                for b in result:result[b].append(ids[b].numpy())
        return {b:np.stack(v).astype(np.int32) for b,v in result.items()}

    def encode_positions(self, values, dataset, native_indices, observed=None):
        if self.metadata.get('global_analysis'):
            raise ValueError('global analysis requires full native montage before selecting local IDs')
        values = np.asarray(values)
        indices = np.broadcast_to(np.asarray(native_indices), values.shape[:2])
        if indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= len(self.roles(dataset))).any():
            raise ValueError('fixed native indices required')
        observed = np.ones(values.shape[:2], bool) if observed is None else np.asarray(observed)
        if observed.shape != values.shape[:2] or observed.dtype != bool:
            raise ValueError('observation-only availability required')
        prediction = self.predict_continuous(values)
        result = {}
        with torch.no_grad():
            for branch in ('S', 'O'):
                ids = np.zeros((*values.shape[:2], 15), np.int32)
                for band in range(5):
                    selected = indices % 5 == band
                    f, c = self.shortlist(prediction[branch][selected].reshape(-1, 400), branch, band)
                    key = f'{branch}_{band}'
                    centers = self.network.centers[key][torch.from_numpy(c)]
                    scores = -torch.sum((torch.from_numpy(f)[:, None]-centers)**2, -1)/self.network.temperature
                    scores += self.network.bias[key][torch.from_numpy(c)]
                    choice = scores.argmax(-1).numpy()
                    ids[selected] = (c[np.arange(len(c)), choice]+1).reshape(-1, 15)
                ids[~observed] = 0
                result[branch] = ids
        return result

    def decode_ids(self, ids, dataset, *, spatial=True, cross_branch=True):
        if spatial and not cross_branch:
            raise ValueError('the declared coupled montage decoder requires both branch synthesis paths')
        if set(ids) != {'S', 'O'}:
            raise ValueError('both branch integer sequences required')
        shape = np.asarray(ids['S']).shape
        count = len(self.roles(dataset))
        for value in ids.values():
            value = np.asarray(value)
            if (value.ndim != 3 or value.shape != shape or value.shape[1:] != (count, 15)
                    or value.dtype.kind not in 'iu' or (value < 0).any() or (value >= self.codebook_size).any()):
                raise ValueError('one in-vocabulary integer per native position and two-second patch required')
        output = []
        with torch.no_grad():
            for n in range(shape[0]):
                codes = {b: torch.tensor(ids[b][n], dtype=torch.long) for b in ('S', 'O')}
                output.append(self.network.decode_ids(codes, dataset, spatial=spatial).numpy())
        values = np.stack(output)
        result=dict(S=values[:,0],O=values[:,1],sum=values[:,0]+values[:,1])
        if spatial and self.phase_readout:
            coefficient=self.phase_readout[dataset]
            if self.metadata['integer_phase_readout']=='cross_electrode_complex_residual':
                raw=np.stack([result[b].reshape(shape[0],count,6000) for b in ('S','O')],axis=-1).transpose(0,2,1,3).reshape(shape[0],6000,2*count)
                spectrum=np.fft.rfft(raw,axis=1,norm='ortho');transformed=np.zeros_like(spectrum)
                bin_hz=self.metadata['phase_bin_hz']
                for k,lo in enumerate(np.arange(1,45,bin_hz)):
                    begin,end=int(lo*30),int(min(lo+bin_hz,45)*30)
                    transformed[:,begin:end]=spectrum[:,begin:end]@coefficient[k]
                wave=np.fft.irfft(transformed,n=6000,axis=1,norm='ortho').reshape(shape[0],6000,count,2).transpose(0,2,1,3)
                result={b:wave[...,k].reshape(shape[0],count,15,400) for k,b in enumerate(('S','O'))}
                result['sum']=result['S']+result['O']
                return result
            if self.metadata['integer_phase_readout']=='wiener_integer_sum':
                spectrum=np.fft.rfft(result['sum'].reshape(shape[0],count,6000),axis=-1,norm='ortho')
                transformed=spectrum[...,None]*coefficient[None,None,:,0,:]
            else:
                spectrum=np.stack([np.fft.rfft(result[b].reshape(shape[0],count,6000),axis=-1,norm='ortho') for b in ('S','O')],axis=-1)
                transformed=np.einsum('bpfj,fjo->bpfo',spectrum,coefficient,optimize=True)
            wave=np.fft.irfft(transformed,n=6000,axis=2,norm='ortho')
            result={b:wave[...,k].reshape(shape[0],count,15,400) for k,b in enumerate(('S','O'))}
            result['sum']=result['S']+result['O']
        return result

    def save(self, path):
        if bool(self.phase_readout)!=bool(self.metadata.get('integer_phase_readout')):
            raise ValueError('phase coefficients and declared decoder mode must agree')
        payload = dict(metadata=np.asarray(json.dumps(self.metadata)))
        for branch in ('S', 'O'):
            payload['transfer_'+branch] = self.transfer[branch]
            for band in range(5):
                for field in ('centers', 'atoms', 'feature_weights'):
                    payload[f'{field}_{branch}_{band}'] = getattr(self, field)[branch, band]
                for dataset in self.metadata['positions']:
                    payload[f'weights_{dataset}_{branch}_{band}'] = self.weights[dataset, branch, band]
        for key, value in self.network.state_dict().items():
            payload['network_'+key] = value.detach().numpy()
        for dataset,value in self.phase_readout.items():payload['phase_readout_'+dataset]=value
        path = Path(path)
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
        if metadata['schema'] not in ('montage_joint_tokenizer_v1','montage_joint_phase_readout_v1'):
            raise ValueError('versioned joint montage model required')
        fields = {name: {(b, k): stored[f'{name}_{b}_{k}'] for b in ('S', 'O') for k in range(5)}
                  for name in ('centers', 'atoms', 'feature_weights')}
        source = SpatialWaveformDictionary(metadata, {b: stored['transfer_'+b] for b in ('S', 'O')},
            weights={(d, b, k): stored[f'weights_{d}_{b}_{k}'] for d in metadata['positions'] for b in ('S', 'O') for k in range(5)}, **fields)
        model = MontageJointTokenizer(source, metadata)
        state = {key.removeprefix('network_'): torch.from_numpy(stored[key]) for key in stored.files if key.startswith('network_')}
        model.network.load_state_dict(state)
        if metadata['schema']=='montage_joint_phase_readout_v1':
            if metadata.get('integer_phase_readout') not in ('wiener_integer_sum','complex2x2','cross_electrode_complex_residual'):
                raise ValueError('known frozen integer phase readout mode required')
            model.phase_readout={d:stored['phase_readout_'+d] for d in metadata['positions']}
            expected={d:((len(np.arange(1,45,metadata['phase_bin_hz'])),2*len(names),2*len(names))
                        if metadata['integer_phase_readout']=='cross_electrode_complex_residual' else (3001,2,2))
                      for d,names in metadata['positions'].items()}
            if any(v.shape!=expected[d] or not np.isfinite(v).all() for d,v in model.phase_readout.items()):
                raise ValueError('complete finite frozen 30s complex readout required')
    model.network.eval()
    return model


def phase_basis_statistics(waves,targets,mask,bin_hz=2):
    """Exact masked time-domain quadratic forms for shared complex Fourier bins."""
    waves=np.asarray(waves,float);targets=np.asarray(targets,float);mask=np.asarray(mask,bool)
    if waves.shape!=(2,6000) or targets.shape!=(3,6000) or mask.shape!=(6000,) or not mask.any():
        raise ValueError('two integer-decoded carriers, three targets and nonempty 30s mask required')
    if not np.isfinite(waves).all() or not np.isfinite(targets[:,mask]).all():raise ValueError('nonfinite phase fit')
    z=np.fft.rfft(waves,axis=-1,norm='ortho');y=np.fft.rfft(np.where(mask,targets,0.),axis=-1,norm='ortho')
    bins=[np.arange(int(lo*30),int(min(lo+bin_hz,45)*30)) for lo in np.arange(1,45,bin_hz)]
    width=4*len(bins);gram=np.zeros((width,width));cross=np.zeros((3,width));invalid=np.flatnonzero(~mask)
    correction=np.zeros((len(invalid),width))
    for band,indices in enumerate(bins):
        values=z[:,indices]
        covariance=2*values.conj()@values.T
        offset=4*band
        for i in range(2):
            for j in range(2):
                value=covariance[i,j]
                gram[offset+2*i:offset+2*i+2,offset+2*j:offset+2*j+2]=[[value.real,-value.imag],[value.imag,value.real]]
            value=2*np.sum(values[i].conj()[None]*y[:,indices],axis=-1)
            cross[:,offset+2*i]=value.real;cross[:,offset+2*i+1]=value.imag
        if len(invalid):
            if len(invalid)<=256:
                phase=np.exp(2j*np.pi*indices[:,None]*invalid[None]/6000)
                analytic=2/np.sqrt(6000)*(values@phase)
                for i in range(2):
                    correction[:,offset+2*i]=analytic[i].real;correction[:,offset+2*i+1]=-analytic[i].imag
            else:
                retained=np.zeros((2,3001),complex);retained[:,indices]=values
                for i in range(2):
                    correction[:,offset+2*i]=np.fft.irfft(retained[i],n=6000,norm='ortho')[invalid]
                    correction[:,offset+2*i+1]=np.fft.irfft(1j*retained[i],n=6000,norm='ortho')[invalid]
    gram-=correction.T@correction
    count=mask.sum()
    return gram/count,cross/count,np.sum(np.where(mask,targets,0.)**2,axis=1)/count


def phase_joint_quadratic(gram,cross,energy,variance):
    """Equal X/S/O training-variance NMSE, with X defined as decoded S+O."""
    inverse=1/np.asarray(variance,float);width=len(gram)
    matrix=np.block([[(inverse[0]+inverse[1])*gram,inverse[0]*gram],
                     [inverse[0]*gram,(inverse[0]+inverse[2])*gram]])
    linear=np.r_[inverse[0]*cross[0]+inverse[1]*cross[1],inverse[0]*cross[0]+inverse[2]*cross[2]]
    return matrix,linear,float(inverse@energy)


def phase_coefficient(vector,bin_hz=2):
    bins=list(np.arange(1,45,bin_hz));part=np.asarray(vector).reshape(2,len(bins),2,2)
    coefficient=np.zeros((3001,2,2),complex)
    for b,lo in enumerate(bins):
        for output in range(2):
            coefficient[int(lo*30):int(min(lo+bin_hz,45)*30),:,output]=part[output,b,:,0]+1j*part[output,b,:,1]
    return coefficient


def fit_phase_quadratic(matrix,linear,bin_hz,ridge,smooth):
    bins=len(np.arange(1,45,bin_hz));width=4*bins;prior=np.zeros((2,bins,2,2))
    for output in range(2):prior[output,:,output,0]=1.
    prior=prior.ravel();scale=max(float(np.trace(matrix))/len(matrix),1e-12)
    regularizer=np.eye(2*width)*ridge*scale
    for output in range(2):
        for b in range(bins-1):
            for coordinate in range(4):
                i=output*width+4*b+coordinate;j=i+4
                regularizer[i,i]+=smooth*scale;regularizer[j,j]+=smooth*scale
                regularizer[i,j]-=smooth*scale;regularizer[j,i]-=smooth*scale
    return np.linalg.solve(matrix+regularizer+np.eye(len(matrix))*1e-12,linear+ridge*scale*prior)


def lowrank_spatial_residual(covariance,cross,prior,rank,ridge):
    """Strong-ridge reduced-rank residual in training-standardized carrier units."""
    count=len(covariance);variance=np.maximum(np.diag(covariance).real,1e-12)
    scale=1/np.sqrt(np.maximum(variance,max(variance.mean()*1e-6,1e-12)))
    normalized=covariance*scale[:,None]*scale[None,:]
    eigenvectors_value,vectors=np.linalg.eigh((normalized+normalized.conj().T)/2)
    values=np.maximum(eigenvectors_value,0.);chosen=np.argsort(values)[-min(rank,count):];v=vectors[:,chosen];retained=values[chosen]
    strength=ridge*max(float(np.trace(normalized).real/count),1e-12)
    remainder=cross-covariance@prior
    left=scale[:,None]*v
    right=(v.conj().T@(scale[:,None]*remainder))/(retained[:,None]+strength)
    positive=values[values>max(values.max()*1e-10,1e-12)]
    diagnostics=dict(numerical_rank=len(positive),raw_positive_condition=float(positive.max()/positive.min()) if len(positive) else None,
        retained_regularized_condition=float((retained.max()+strength)/(retained.min()+strength)),
        shrinkage=(retained/(retained+strength)).tolist(),effective_rank=float(np.sum(retained/(retained+strength))),ridge_strength=strength)
    return left,right,diagnostics
