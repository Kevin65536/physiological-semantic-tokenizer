"""Fixed anatomical EEG neighborhoods and paired Hb allocation diagnostics."""
from collections import defaultdict
import numpy as np
from scipy.signal import butter,sosfiltfilt

BANDS=((1,4),(4,8),(8,13),(13,30),(30,45))


def fixed_neighborhoods(records,positions):
    found=defaultdict(set);counts=defaultdict(int);systems=defaultdict(set)
    for record in records:
        if record['split']!='train':continue
        dataset=record['dataset']
        for modality,rows in record['geometry_identity'].items():
            if any(r.get('measured_subject_coordinate') is True for r in rows):
                raise ValueError('fixed dataset template required, no individual coordinate input')
        names=record['eeg_channel_names']
        for pair in record['pairs']:
            selected=pair['channel_sets']['nearest6'];channels=tuple(names[i] for i in selected)
            if len(set(channels))!=6 or list(channels)!=pair['channel_set_names']['nearest6']:
                raise ValueError('exact owning nearest6 identities required')
            found[dataset,pair['id']].add(channels);counts[dataset]+=1
            systems[dataset].add((pair['coordinate_system'],pair['coordinate_units']))
    result={};controls={};overlaps={}
    for dataset,sites in positions.items():
        result[dataset]={}
        for site in sites:
            values=found[dataset,site]
            if len(values)!=1:raise ValueError('one fixed train metadata neighborhood per deployed Hb position required')
            result[dataset][site]=list(next(iter(values)))
        order=sorted(sites);shift=len(order)//2
        controls[dataset]={site:order[(i+shift)%len(order)] for i,site in enumerate(order)}
        overlaps[dataset]=float(np.mean([len(set(result[dataset][site])&set(result[dataset][controls[dataset][site]]))/6 for site in order]))
    return dict(nearest6=result,cyclic_site=controls,mean_cyclic_overlap=overlaps,
        coordinate_systems={d:[list(v) for v in sorted(s)] for d,s in systems.items()},
        rule='exact unique train-template nearest6; cyclic half-rotation of sorted fixed native Hb names',individual_PC_or_target_values_used=False)


def power_bank(values,observed):
    values=np.asarray(values,float);observed=np.asarray(observed,bool)
    if values.ndim!=2 or values.shape[0]!=6000 or observed.shape!=(values.shape[1],):
        raise ValueError('same visible30s EEG200Hz and native availability required')
    spectrum=np.fft.rfft(values,axis=0);hz=np.fft.rfftfreq(6000,1/200);out=np.zeros((300,values.shape[1],5))
    for k,(lo,hi) in enumerate(BANDS):
        keep=(hz>=lo)&((hz<=hi) if hi==45 else (hz<hi));wave=np.fft.irfft(spectrum*keep[:,None],n=6000,axis=0)
        out[:,:,k]=(wave.reshape(300,20,-1)**2).mean(1)
    out[:,~observed]=0;return out


def pool_power(power,observed,indices):
    indices=np.asarray(indices,int);indices=indices[np.asarray(observed,bool)[indices]]
    if not len(indices):return np.zeros((300,10)),np.zeros(300,bool)
    p=power[:,indices];absolute=np.log1p(p).mean(1)
    relative=np.divide(p,p.sum(-1,keepdims=True),out=np.zeros_like(p),where=p.sum(-1,keepdims=True)>1e-20).mean(1)
    return np.c_[absolute,relative],np.ones(300,bool)


def lags(values,valid,seconds):
    values=np.asarray(values,float);valid=np.asarray(valid,bool)
    if values.shape[0]!=300 or valid.shape!=(300,):raise ValueError('same30s 10Hz features and availability required')
    xs=[];masks=[]
    for lag in seconds:
        shift=int(lag*10);x=np.zeros_like(values);mask=np.zeros(300,bool)
        if abs(shift)>=300:raise ValueError('lag outside visible context')
        if shift>0:x[shift:]=values[:-shift];mask[shift:]=valid[:-shift]
        elif shift<0:x[:shift]=values[-shift:];mask[:shift]=valid[-shift:]
        else:x[:]=values;mask[:]=valid
        xs.append(np.where(mask[:,None],x,0.));masks.append(mask)
    return np.concatenate(xs,axis=1),np.stack(masks,axis=1).astype(float)


def design(hb,base,eeg,hb_valid,eeg_valid,position,npositions,seconds):
    h=np.c_[hb,base['S'],base['O']];x,valid=lags(h,hb_valid,seconds);parts=[x,valid,np.broadcast_to(np.eye(npositions)[position],(300,npositions))]
    if eeg is not None:
        e,seen=lags(eeg,eeg_valid,seconds)
        parts.extend([e,seen,(e[:,:,None]*hb[:,None,:]).reshape(300,-1)])
    raw=np.concatenate(parts,axis=1)
    return sosfiltfilt(butter(4,.4,fs=10,output='sos'),raw,axis=0).astype(np.float32)


def fit_ridge(gram,right,weight_sum,ridge):
    g=np.asarray(gram,float)/weight_sum;r=np.asarray(right,float)/weight_sum
    scale=np.sqrt(np.maximum(np.diag(g),1e-16));normalized=g/scale[:,None]/scale[None]
    coefficient=np.linalg.solve(normalized+ridge*np.eye(len(scale)),r/scale)/scale
    return coefficient,scale


def allocate(base,delta,quiet):
    delta=np.asarray(delta,float).copy();delta[np.asarray(quiet,bool)]=0
    result=dict(S=np.asarray(base['S'],float)-delta,O=np.asarray(base['O'],float)+delta,sum=np.asarray(base['sum']).copy())
    for b in result:result[b][quiet]=base[b][quiet]
    return result
