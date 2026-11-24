"""Hard source-attribute routing followed by one paid native-band waveform atom."""
import json
from pathlib import Path
import numpy as np
from .spectral_waveform_dictionary import raw_features

BANDS=((1,4),(4,8),(8,13),(13,30),(30,45))


def bins(samples,band):
    hz=np.fft.rfftfreq(samples,1/200);lo,hi=BANDS[band]
    return np.flatnonzero((hz>=lo)&((hz<=hi) if hi==45 else (hz<hi)))


def project(values,band):
    values=np.asarray(values,float);spectrum=np.fft.rfft(values,axis=-1)
    keep=np.zeros_like(spectrum);keep[...,bins(values.shape[-1],band)]=spectrum[...,bins(values.shape[-1],band)]
    return np.fft.irfft(keep,n=values.shape[-1],axis=-1)


def coordinates(values,band):
    z=np.fft.rfft(np.asarray(values,float),axis=-1,norm='ortho')[...,bins(400,band)]
    return np.concatenate((z.real,z.imag),axis=-1)*np.sqrt(2)


def inverse(values,band):
    n=len(bins(400,band));z=np.zeros((len(values),201),complex)
    z[:,bins(400,band)]=(values[:,:n]+1j*values[:,n:])/np.sqrt(2)
    return np.fft.irfft(z,n=400,axis=-1,norm='ortho')


def route(features,tree):
    features=np.asarray(features,float);result=np.empty(len(features),np.int32)
    def descend(node,indices):
        if 'leaf' in node:result[indices]=node['leaf'];return
        left=features[indices,node['axis']]<=node['threshold']
        descend(node['left'],indices[left]);descend(node['right'],indices[~left])
    descend(tree,np.arange(len(features)));return result


def fit_tree(features,max_depth=3):
    leaves=[]
    def split(indices,depth):
        if depth==max_depth or len(indices)<128:
            node=dict(leaf=len(leaves));leaves.append(indices);return node
        # Each path explicitly uses theta, alpha and RMS, never a soft weight.
        axis=depth%3;values=features[indices,axis];unique=np.unique(values)
        if len(unique)<2:
            node=dict(leaf=len(leaves));leaves.append(indices);return node
        median=np.median(values);j=int(np.searchsorted(unique,median,side='right'))
        j=min(max(j,1),len(unique)-1);threshold=float((unique[j-1]+unique[j])/2)
        left=values<=threshold
        if min(left.sum(),(~left).sum())<64:
            node=dict(leaf=len(leaves));leaves.append(indices);return node
        return dict(axis=axis,threshold=threshold,left=split(indices[left],depth+1),right=split(indices[~left],depth+1))
    return split(np.arange(len(features)),0),leaves


def nearest(features,centers):
    ids=np.empty(len(features),np.int32);norm=(centers**2).sum(1)
    for start in range(0,len(features),256):
        q=features[start:start+256];d=(q*q).sum(1)[:,None]+norm-2*q@centers.T
        ids[start:start+len(q)]=d.argmin(1)
    return ids


def fit_book(physical,band,*,positive_atoms=511,seed=20261123,max_iter=10):
    from sklearn.cluster import MiniBatchKMeans
    physical=np.asarray(physical,float).reshape(-1,400)
    semantic=raw_features(physical,None,None,mode='theta_alpha_rms')
    tree,groups=fit_tree(semantic);feature=coordinates(physical,band)
    capacities=np.full(len(groups),positive_atoms//len(groups));capacities[:positive_atoms%len(groups)]+=1
    centers=[];atoms=[np.zeros(400)];offset=[1];counts=[];reports=[]
    for group,(selected,size) in enumerate(zip(groups,capacities)):
        if len(selected)<size:raise ValueError('hard group has fewer real patches than atoms')
        clustering=MiniBatchKMeans(n_clusters=int(size),n_init=1,random_state=seed+group,
            max_iter=max_iter,batch_size=1024,max_no_improvement=8,reassignment_ratio=0.)
        q=feature[selected];labels=clustering.fit_predict(q)
        count=np.bincount(labels,minlength=int(size));center=clustering.cluster_centers_.astype(float)
        # Recompute final Euclidean conditional means; dead atoms retain a real
        # training example nearest the fitted center, never a teacher waveform.
        center=np.zeros_like(center);np.add.at(center,labels,q);live=count>0
        center[live]/=count[live,None]
        for dead in np.flatnonzero(~live):center[dead]=q[np.argmin(((q-clustering.cluster_centers_[dead])**2).sum(1))]
        centers.append(center);atoms.extend(inverse(center,band));counts.extend(count.tolist());offset.append(offset[-1]+int(size))
        reports.append(dict(group=group,training_patches=len(selected),atoms=int(size),occupied=int(live.sum()),iterations=int(clustering.n_iter_)))
    return dict(tree=tree,centers=np.vstack(centers),atoms=np.array(atoms),offset=np.array(offset,np.int32),counts=np.r_[0,counts],report=reports)


def assign_book(physical,band,book):
    physical=np.asarray(physical,float).reshape(-1,400)
    group=route(raw_features(physical,None,None,mode='theta_alpha_rms'),book['tree'])
    q=coordinates(physical,band);ids=np.empty(len(q),np.int32)
    for leaf,(lo,hi) in enumerate(zip(book['offset'][:-1],book['offset'][1:])):
        selected=np.flatnonzero(group==leaf)
        ids[selected]=lo+nearest(q[selected],book['centers'][lo-1:hi-1])
    return ids


def solve_readout(gram,cross,variance,ridge=.01):
    """Same-rank native readout fitted jointly to true S, O and their true X sum.

    cross is [target_site,source_slot,(S,O,X)]. The fixed train variances
    determine the 2x2 output metric, including the X reconstruction term.
    """
    gram=np.asarray(gram,float);cross=np.asarray(cross,float);variance=np.asarray(variance,float)
    count,width,_=gram.shape;weights=np.zeros((width,count,2))
    for p in range(count):
        vs,vo,vx=variance[p]
        h=np.array([[1/vs+1/vx,1/vx],[1/vx,1/vo+1/vx]])
        rhs=np.column_stack((cross[p,:,0]/vs+cross[p,:,2]/vx,cross[p,:,1]/vo+cross[p,:,2]/vx))@np.linalg.inv(h)
        penalty=max(1e-12,ridge*np.trace(gram[p])/width)
        weights[:,p]=np.linalg.solve(gram[p]+penalty*np.eye(width),rhs)
    return weights


class StratifiedSourceDictionary:
    def __init__(self,metadata,books,weights):
        self.metadata=metadata;self.books=books;self.weights=weights;self.codebook_size=metadata['codebook_size']
        if self.codebook_size!=512 or metadata['schema']!='stratified_source_dictionary_v1':raise ValueError('versioned K512 atomic dictionary required')
        for b in 'SO':
            for k in range(5):
                book=books[b,k];offset=np.asarray(book['offset'])
                if (book['atoms'].shape!=(512,400) or book['centers'].shape!=(511,2*len(bins(400,k)))
                    or book['counts'].shape!=(512,) or offset.ndim!=1 or offset[0]!=1 or offset[-1]!=512 or np.any(np.diff(offset)<=0)
                    or not np.isfinite(book['atoms']).all() or not np.isfinite(book['centers']).all() or np.any(book['atoms'][0]!=0)):
                    raise ValueError('finite atoms, role Fourier centers, hard routing offsets and reserved zero required')
        if weights:
            for d,names in metadata['positions'].items():
                for b in 'SO':
                    for k in range(5):
                        w=weights[d,b,k]
                        if w.shape!=(2*int((self.roles(d)==k).sum()),len(names)) or not np.isfinite(w).all():raise ValueError('complete fixed native dual-branch spatial readout required')

    def roles(self,dataset):return np.arange(len(self.metadata['positions'][dataset]))%5

    def assign(self,physical,dataset,observed):
        roles=self.roles(dataset);shape=np.shape(physical['S']);observed=np.asarray(observed)
        if len(shape)!=4 or shape[1:]!=(len(roles),15,400) or np.shape(physical['O'])!=shape or observed.shape!=shape[:2] or observed.dtype!=bool:
            raise ValueError('full native 30s source and observation flags required')
        output={}
        for branch in 'SO':
            code=np.zeros(shape[:3],np.int16)
            for band in range(5):
                sites=np.flatnonzero(roles==band)
                code[:,sites]=assign_book(physical[branch][:,sites],band,self.books[branch,band]).reshape(shape[0],len(sites),15)
            code[~observed]=0;output[branch]=code
        return output

    def atom_carriers(self,ids,dataset):
        if set(ids)!=set('SO'):raise ValueError('both integer branches required')
        roles=self.roles(dataset);carriers={};shape=None
        for branch in 'SO':
            code=np.asarray(ids[branch])
            if code.ndim!=3 or code.shape[1:]!=(len(roles),15) or code.dtype.kind not in 'iu' or np.any(code<0) or np.any(code>=512):raise ValueError('one integer per native site and 2s required')
            if shape is not None and code.shape!=shape:raise ValueError('matching branch support required')
            shape=code.shape;wave=np.zeros((*shape,400))
            for band in range(5):
                sites=np.flatnonzero(roles==band);wave[:,sites]=self.books[branch,band]['atoms'][code[:,sites]]
            carriers[branch]=wave
        return carriers

    def synthesize(self,carriers,dataset):
        roles=self.roles(dataset);shape=np.shape(carriers['S']);output={b:np.zeros(shape) for b in 'SO'}
        for band in range(5):
            sites=np.flatnonzero(roles==band)
            features=np.concatenate([project(np.asarray(carriers[b])[:,sites].reshape(shape[0],len(sites),6000),band) for b in 'SO'],axis=1)
            for b in 'SO':output[b]+=(np.einsum('bmt,mc->bct',features,self.weights[dataset,b,band],optimize=True)).reshape(shape)
        output['sum']=output['S']+output['O'];return output

    def decode_ids(self,ids,dataset):return self.synthesize(self.atom_carriers(ids,dataset),dataset)

    def save(self,path):
        payload=dict(metadata=json.dumps(self.metadata))
        for (b,k),book in self.books.items():
            for name in ['centers','atoms','offset','counts']:payload[f'{b}_{k}_{name}']=book[name]
            payload[f'{b}_{k}_tree']=json.dumps(book['tree'])
        for (d,b,k),value in self.weights.items():payload[f'weights_{d}_{b}_{k}']=value
        np.savez_compressed(path,**payload)


def load_model(path):
    with np.load(Path(path)) as saved:
        metadata=json.loads(str(saved['metadata']));books={}
        for b in 'SO':
            for k in range(5):books[b,k]=dict(tree=json.loads(str(saved[f'{b}_{k}_tree'])),**{name:saved[f'{b}_{k}_{name}'].copy() for name in ['centers','atoms','offset','counts']})
        weights={(d,b,k):saved[f'weights_{d}_{b}_{k}'].copy() for d in metadata['positions'] for b in 'SO' for k in range(5)}
    return StratifiedSourceDictionary(metadata,books,weights)


def pack_ids(values):
    values=np.asarray(values)
    if values.ndim!=1 or values.dtype.kind not in 'iu' or np.any(values<0) or np.any(values>=512):raise ValueError('K512 integer vector required')
    return np.packbits(((values[:,None]>>np.arange(9))&1).astype(np.uint8).reshape(-1),bitorder='little').tobytes()


def unpack_ids(payload,count):
    if len(payload)!=(count*9+7)//8:raise ValueError('exact 9bit payload length required')
    bits=np.unpackbits(np.frombuffer(payload,np.uint8),bitorder='little')
    if bits[count*9:].any():raise ValueError('nonzero padding')
    return (bits[:count*9].reshape(count,9)*2**np.arange(9)).sum(1).astype(np.int16)
