"""One paid pilot or conditional waveform word at each native EEG branch and slot."""
from __future__ import annotations

import numpy as np

from .band_event_dictionary import fit_basis
from .band_exchange_words import ATOMS, nearest, paid_coordinates
from .spatial_waveform_dictionary import BANDS
from sklearn.cluster import MiniBatchKMeans

PILOT_SIZE=5116
VOCABULARY=10231


def pilot_positions(n):
    return np.concatenate((np.arange(n)%2==0,np.arange(n)%2==1))


def condition_on_pilots(covariance,roles,gain,noise):
    n=len(roles)//2;pilot=pilot_positions(n)
    operators=[];conditional=[]
    for band,C in enumerate(covariance):
        positions=np.flatnonzero((roles==band)&pilot)
        g=gain[band,positions];r=noise[band,positions]
        measured=g[:,None]*C[np.ix_(positions,positions)]*g[None,:]+np.diag(r)
        good=np.diag(measured)>1e-20
        positions=positions[good];g=g[good]
        if len(positions):
            mapping=np.linalg.solve(measured[np.ix_(good,good)],g[:,None]*C[positions]).T
            after=C-(mapping*g[None,:])@C[positions]
        else:
            mapping=np.empty((len(roles),0));after=C.copy()
        after=(after+after.T)/2
        operators.append((positions,mapping))
        conditional.append(after)
    return operators,np.asarray(conditional)


def posterior_waveforms(pilot_coordinates,second_coordinates,roles,covariance,pilot_gain,pilot_noise,second_gain,second_noise,mode):
    if mode not in ("innovation","direct"):
        raise ValueError("registered conditional or direct observation required")
    n=len(roles)//2;pilot=pilot_positions(n)
    operators,conditional=condition_on_pilots(covariance,roles,pilot_gain,pilot_noise)
    result=np.zeros((2*n,15,400))
    for band,(P,mapping) in enumerate(operators):
        mean=(mapping@pilot_coordinates[P].reshape(len(P),6000)).reshape(2*n,15,400)
        J=np.flatnonzero((roles==band)&~pilot)
        C=conditional[band]
        g=second_gain[band,J]
        measured=g[:,None]*C[np.ix_(J,J)]*g[None,:]+np.diag(second_noise[band,J])
        good=np.diag(measured)>1e-20
        if good.any():
            positions=J[good];g=g[good]
            operator=np.linalg.solve(measured[np.ix_(good,good)],g[:,None]*C[positions]).T
            observed=second_coordinates[positions].reshape(len(positions),6000)
            if mode=="direct":observed=observed-g[:,None]*mean[positions].reshape(len(positions),6000)
            mean+=(operator@observed).reshape(2*n,15,400)
        result+=mean
    wave=result.reshape(2*n,6000);spectrum=np.fft.rfft(wave)
    hz=np.fft.rfftfreq(6000,1/200);spectrum[:,(hz<1)|(hz>45)]=0
    s,o=np.fft.irfft(spectrum,n=6000).reshape(2,n,15,400)
    return dict(S=s,O=o,sum=s+o)


def pilot_means(coordinates,roles,covariance,pilot_gain,pilot_noise):
    operators,conditional=condition_on_pilots(covariance,roles,pilot_gain,pilot_noise)
    result=[]
    for positions,mapping in operators:
        result.append((mapping@coordinates[positions].reshape(len(positions),6000)).reshape(coordinates.shape))
    return np.asarray(result),conditional


def fit_second_books(waves,attributes,source_energy,weights,role,seed,capacity=ATOMS):
    """Fit two equal-budget books to normalized direct and innovation carriers."""
    properties=np.asarray(attributes,np.float32);weights=np.asarray(weights,float)
    weights/=weights.mean();count=max(1,min(capacity,len(properties)//8))
    property_mean=np.average(properties,axis=0,weights=weights).astype(np.float32)
    property_scale=np.sqrt(np.average(np.square(properties-property_mean),axis=0,weights=weights)).clip(1e-7).astype(np.float32)
    energy=np.asarray(source_energy,float);capped=np.minimum(energy,np.quantile(energy,.99))
    probability=.5*weights/weights.sum()+.5*weights*capped/max(np.sum(weights*capped),1e-20)
    probability/=probability.sum()
    initial=np.random.default_rng(seed).choice(len(properties),count,replace=False,p=probability)
    output=dict(capacity=np.int32(capacity),property_mean=property_mean,property_scale=property_scale)
    width=min(76,4*(BANDS[role][1]-BANDS[role][0])+8,len(properties))
    for mode,values in waves.items():
        y=np.asarray(values,np.float32)
        mean,basis,fraction=fit_basis(y,weights,width)
        projection=(y-mean)@basis.T
        wave_scale=max(np.sqrt(np.average(np.square(projection).sum(1),weights=weights)),1e-8)
        coordinates=np.column_stack((projection/np.float32(wave_scale),(properties-property_mean)/(property_scale*np.float32(np.sqrt(3))))).astype(np.float32)
        model=MiniBatchKMeans(n_clusters=count,init=coordinates[initial].copy(),n_init=1,max_iter=30,max_no_improvement=8,batch_size=2048,reassignment_ratio=0,random_state=seed)
        model.fit(coordinates,sample_weight=weights.astype(np.float32));centers=model.cluster_centers_.astype(np.float32)
        ids=nearest(coordinates,centers);mass=np.bincount(ids,weights=weights,minlength=count)
        atoms=np.zeros((count,400));np.add.at(atoms,ids,y*weights[:,None]);atoms/=mass.clip(1e-20)[:,None]
        empty=mass==0
        if empty.any():atoms[empty]=y[nearest(centers[empty],coordinates)]
        full_energy=np.square(y.astype(float)).sum(1)
        expected=np.bincount(ids,weights=weights*full_energy,minlength=count)/mass.clip(1e-20)
        gain=np.sqrt(expected/np.square(atoms).sum(1).clip(1e-14)).clip(1,2)
        output.update({mode+"_mean":mean,mode+"_basis":basis,mode+"_wave_scale":np.float32(wave_scale),mode+"_centers":centers,
                       mode+"_atoms":atoms.astype(np.float32),mode+"_energy_gain":gain.astype(np.float32),mode+"_count":np.bincount(ids,minlength=count),mode+"_PCA_fraction":np.array(fraction)})
    return output


def encode_second(values,properties,book,mode,role):
    projected=((values-book[mode+"_mean"])@book[mode+"_basis"].T)/book[mode+"_wave_scale"]
    feature=(properties-book["property_mean"])/(book["property_scale"]*np.float32(np.sqrt(3)))
    ids=nearest(np.column_stack((projected,feature)).astype(np.float32),book[mode+"_centers"])
    return (PILOT_SIZE+role*int(book.get("capacity",ATOMS))+ids).astype(np.uint16)


def paid_fields(sent,old_books,new_books,scale,mode,energy):
    n=len(sent["S"]);pilot=pilot_positions(n)
    first=np.zeros((2*n,15,400));second=np.zeros_like(first);roles=[]
    for bi,b in enumerate("SO"):
        strides={int(book.get("capacity",ATOMS)) for book in new_books[b]}
        if len(strides)!=1:raise ValueError("inconsistent second-word strides")
        stride,=strides
        values=np.asarray(sent[b])
        if values.shape!=(n,15) or values.dtype.kind not in "ui" or np.any(values<0) or np.any(values>=PILOT_SIZE+5*stride):
            raise ValueError("one valid paid word per branch site and two-second slot required")
        stage=values>=PILOT_SIZE
        expected=np.broadcast_to(~pilot[bi*n:(bi+1)*n,None],values.shape)&(values!=0)
        if not np.array_equal(stage,expected):raise ValueError("word type violates the fixed native pilot order")
        old=np.where(stage,0,values)
        decoded,tags=paid_coordinates(old,old_books[b],"balanced",True)
        first[bi*n:(bi+1)*n]=decoded
        tag=tags.copy()
        tag[stage]=(values[stage].astype(np.int32)-PILOT_SIZE)//stride
        if not np.all(tag==tag[:,:1]):raise ValueError("fixed native frequency assignment required")
        roles.append(tag[:,0])
        block=second[bi*n:(bi+1)*n]
        for role,book in enumerate(new_books[b]):
            take=stage&(tag==role);idx=(values[take].astype(np.int32)-PILOT_SIZE)%stride
            atoms=book[mode+"_atoms"]
            active=idx<len(atoms);wave=np.zeros((len(idx),400));wave[active]=atoms[idx[active]]
            if energy:wave[active]*=book[mode+"_energy_gain"][idx[active],None]
            site=np.nonzero(take)[0];block[take]=wave*scale[role,bi*n+site,None]
    if not np.array_equal(roles[0]<0,roles[1]<0):
        raise ValueError("paired native sites have inconsistent missing observations")
    return first,second,np.concatenate(roles)


def decode_words(sent,old_books,new_books,state,calibration,mode,energy):
    first,second,roles=paid_fields(sent,old_books,new_books,state["scale"],mode,energy)
    return posterior_waveforms(first,second,roles,state["joint"],state["pilot_gain"],state["pilot_noise"],
                               calibration[mode+("_energy" if energy else "_mean")+"_gain"],
                               calibration[mode+("_energy" if energy else "_mean")+"_noise"],mode)


class InnovationEEG:
    """Frozen native montage codec, with one paid pilot or residual word at every slot."""

    def __init__(self, metadata, sources, pilot_books, residual_books, states):
        self.metadata = metadata
        self.sources = sources
        self.pilot_books = pilot_books
        self.residual_books = residual_books
        self.states = states

    def _encode_array(self, x, dataset, seen):
        from .adaptive_band_allocation import band_waves
        from .cross_branch_sampling import encode_role, layouts
        from .band_exchange_words import source_properties

        n = len(seen)
        state = self.states[dataset]
        predicted = self.sources[dataset].predict_continuous(x[None], dataset, seen[None])
        role = layouts(seen)['staggered']
        pilot = pilot_positions(n)
        coordinates, words, selected, properties = [], {}, {}, {}
        for b in 'SO':
            waveform = predicted[b][0]
            bands = band_waves(np.fft.rfft(waveform.reshape(n,6000)))
            properties[b] = source_properties(waveform)
            words[b], selected[b] = encode_role(bands, role[b], properties[b], self.pilot_books[b])
            first, _ = paid_coordinates(words[b], self.pilot_books[b], 'balanced', True)
            coordinates.append(first)
        first = np.concatenate(coordinates)
        first[~pilot] = 0
        means, _ = pilot_means(first, np.r_[role['S'],role['O']], state['joint'], state['pilot_gain'], state['pilot_noise'])
        for bi,b in enumerate('SO'):
            for band,book in enumerate(self.residual_books[b]):
                sites = np.flatnonzero((role[b]==band) & ~pilot[bi*n:(bi+1)*n])
                if not len(sites):
                    continue
                joint_sites = bi*n+sites
                residual = (selected[b][sites]-means[band,joint_sites]).astype(np.float32)
                normalized = residual / state['scale'][band,joint_sites,None,None]
                words[b][sites] = encode_second(normalized.reshape(-1,400), properties[b][sites].reshape(-1,3),book,'innovation',band).reshape(len(sites),15)
        return words

    def encode(self, values, dataset, observed):
        x, seen = np.asarray(values), np.asarray(observed)
        n = len(self.metadata['positions'][dataset])
        if x.ndim!=4 or x.shape[1:]!=(n,15,400) or not len(x) or seen.shape!=x.shape[:2] or seen.dtype!=bool:
            raise ValueError('native montage, observation flags and 30 seconds of EEG required')
        words = [self._encode_array(v,dataset,m) for v,m in zip(x,seen)]
        return {b:np.stack([case[b] for case in words]) for b in 'SO'}

    def decode_ids(self, ids, dataset):
        n = len(self.metadata['positions'][dataset])
        if set(ids)!={'S','O'}:
            raise ValueError('both EEG integer arrays required')
        words={b:np.asarray(ids[b]) for b in 'SO'}
        if words['S'].ndim!=3 or words['S'].shape[1:]!=(n,15) or not len(words['S']) or words['S'].shape!=words['O'].shape:
            raise ValueError('S and O integers must share native montage and slot axes')
        state=self.states[dataset]
        predictions=[decode_words({b:words[b][i] for b in 'SO'},self.pilot_books,self.residual_books,state,state,'innovation',False) for i in range(len(words['S']))]
        return {b:np.stack([case[b] for case in predictions]) for b in ('S','O','sum')}


def load_model(path):
    import json
    from .eeg_transient_gain import BoundedTransientSource

    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata']))
        if metadata['schema']!='cross_branch_innovation_eeg_v1':
            raise ValueError('frozen pilot and residual word model required')
        def fields(prefix):
            return {k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}
        def books(name):
            return {b:[fields(f'{name}_{b}_{j}_') for j in range(5)] for b in 'SO'}
        sources,states={},{}
        for ds in metadata['positions']:
            sources[ds]=BoundedTransientSource({b:z[f'{ds}_transfer_{b}'] for b in 'SO'},
                {ds:fields(ds+'_source_param_')},json.loads(str(z[ds+'_source_metadata'])))
            states[ds]=fields(ds+'_state_')
        return InnovationEEG(metadata,sources,books('pilot'),books('second'),states)
