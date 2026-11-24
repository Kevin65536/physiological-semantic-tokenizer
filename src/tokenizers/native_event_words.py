"""Direct native pulses and oscillations can replace a single paid montage word."""
import numpy as np

from .cross_branch_innovations import pilot_positions
from .cross_branch_sampling import layouts
from .montage_word_search import project, synthesis_matrix

FIRST=15351
FREQUENCIES=np.arange(2,91)
LEVELS=2**np.linspace(-4,6,16)
PULSE_BASE=FIRST+len(FREQUENCIES)*8*len(LEVELS)
K=PULSE_BASE+100*5*2*2*len(LEVELS)


def pulse_shapes():
    shapes=[]
    for center in range(0,400,4):
        for width in (3,6,12,24,48):
            u=(np.arange(400)-center)/width
            bell=np.exp(-.5*u*u)
            for raw in ((1-u*u)*bell,u*bell):
                centered=raw-raw.mean()
                shapes.append(centered/np.max(np.abs(centered)))
    return np.array(shapes)


def event_wave(word, pulses):
    code=np.asarray(word,dtype=np.int32)
    if code.ndim!=1 or np.any((code<FIRST)|(code>=K)):raise ValueError('invalid event code')
    out=np.empty((len(code),400))
    harmonic=code<PULSE_BASE
    local=code[harmonic]-FIRST
    bins=FREQUENCIES[local//128];phase=(local//16)%8
    out[harmonic]=LEVELS[local%16,None]*np.cos(2*np.pi*(bins[:,None]*np.arange(400)/400+phase[:,None]/8))
    local=code[~harmonic]-PULSE_BASE
    out[~harmonic]=LEVELS[local%16,None]*np.where((local//16)%2==0,1.,-1.)[:,None]*pulses[local//32]
    return out


def closest_event(target, pulses, norms):
    """Search every signed pulse and all frequency/phase/level combinations."""
    fft=np.fft.rfft(target)[:,FREQUENCIES]
    phase=np.round(np.angle(fft)*4/np.pi).astype(int)%8
    dots=(fft*np.exp(-1j*np.pi*phase/4)).real
    best=dots.argmax(1);row=np.arange(len(best))
    corr=dots[row,best];level=np.abs(corr[:,None]/200-LEVELS).argmin(1)
    benefit=2*corr*LEVELS[level]-200*LEVELS[level]**2
    word=FIRST+best*128+phase[row,best]*16+level
    dot=target@pulses.T
    amp=abs(dot)/norms
    # The nearest amplitude in ordinary coordinates, not nearest in log amplitude.
    level=np.searchsorted(.5*(LEVELS[:-1]+LEVELS[1:]),amp)
    powers=2*abs(dot)*LEVELS[level]-norms*LEVELS[level]**2
    best=powers.argmax(1);replace=powers[row,best]>benefit
    alternative=PULSE_BASE+best*32+(dot[row,best]<0).astype(int)*16+level[row,best]
    return np.where(replace,alternative,word).astype(np.uint16),np.where(replace,powers[row,best],benefit)


def components(sent, table, state, scales, pulses):
    n=len(sent['S']);codes=np.concatenate([sent[b] for b in 'SO'])
    if codes.shape!=(2*n,15) or codes.dtype.kind not in 'iu' or np.any((codes<0)|(codes>=K)):
        raise ValueError('two valid paid arrays required')
    seen=(codes!=0).all(1)
    if not np.array_equal(seen,(codes!=0).any(1)) or not np.array_equal(seen[:n],seen[n:]):
        raise ValueError('inconsistent native observation fields')
    tags=layouts(seen[:n])['staggered'];roles=np.r_[tags['S'],tags['O']];pilots=pilot_positions(n)
    matrix=synthesis_matrix(roles,state);atoms=np.zeros((2*n,15,400));local=np.zeros_like(atoms)
    for j in np.flatnonzero(seen):
        book=table['SO'[j//n],int(roles[j]),bool(pilots[j])]
        event=codes[j]>=FIRST
        index=codes[j,~event].astype(int)-book['offset']
        if np.any((index<0)|(index>=len(book['atoms']))):raise ValueError('ordinary ID does not match native role or frozen book')
        atoms[j,~event]=book['atoms'][index]
        local[j,event]=scales[j]*event_wave(codes[j,event],pulses)
    unfiltered=(matrix@atoms.reshape(2*n,-1)).reshape(3*n,15,400)
    unfiltered[:2*n]+=local
    unfiltered[2*n:]+=local[:n]+local[n:]
    return matrix,atoms,project(unfiltered)


def decode_events(sent,table,state,scales,pulses):
    _,_,result=components(sent,table,state,scales,pulses);n=len(sent['S'])
    return dict(S=result[:n],O=result[n:2*n],sum=result[:n]+result[n:2*n])


def replace_events(sent,table,state,scales,target,weight,pulses,cap,rounding="ceil"):
    """Rank single-token improvements, then accept only true decreases of a majorant."""
    matrix,atoms,start=components(sent,table,state,scales,pulses)
    n=len(sent['S']);rows=np.arange(2*n);total_sites=2*n+rows%n
    metric=(weight[rows]+weight[total_sites])*scales**2
    gram=np.einsum('ij,i,ij->j',matrix,weight,matrix)
    overlap=scales*(weight[rows]*matrix[rows,rows]+weight[total_sites]*matrix[total_sites,rows])
    error=project(target-start)
    norms=(pulses**2).sum(1)
    local_gradient=scales[:,None,None]*(weight[rows,None,None]*error[rows]+weight[total_sites,None,None]*error[total_sites])
    desired=(local_gradient+overlap[:,None,None]*atoms)/np.maximum(metric,1e-30)[:,None,None]
    old_gradient=(matrix.T@(weight[:,None]*error.reshape(3*n,-1))).reshape(2*n,15,400)
    removal=gram[:,None]*(atoms**2).sum(2)+2*(old_gradient*atoms).sum(2)
    words=np.zeros((2*n,15),np.uint16);gain=np.zeros((2*n,15))
    for j in np.flatnonzero(metric>0):
        words[j],advantage=closest_event(desired[j],pulses,norms)
        gain[j]=metric[j]*advantage-removal[j]
    result=np.concatenate([sent[b] for b in 'SO']).copy()
    if rounding not in ("ceil","floor"):raise ValueError("invalid paid event budget rounding")
    round_limit=np.floor if rounding=="floor" else np.ceil
    totals=[0,0];maximum=[int(round_limit(cap*np.count_nonzero(result[k*n:(k+1)*n]))) for k in range(2)]
    weighted=error*weight[:,None,None]
    for index in np.argsort(-gain.ravel(),kind='stable'):
        j,slot=divmod(int(index),15);b=j//n
        if gain[j,slot]<=0 or totals==maximum:break
        if totals[b]>=maximum[b]:continue
        new=event_wave(words[j,slot:slot+1],pulses)[0]*scales[j]
        delta=-matrix[:,j,None]*atoms[j,slot]
        delta[j]+=new;delta[total_sites[j]]+=new
        change=np.sum(weight[:,None]*delta**2-2*weighted[:,slot]*delta)
        if change>=0:continue
        result[j,slot]=words[j,slot];weighted[:,slot]-=weight[:,None]*delta;totals[b]+=1
    new_words={'S':result[:n],'O':result[n:]}
    after=decode_events(new_words,table,state,scales,pulses)
    loss0=float(np.sum(weight[:,None,None]*(start-target)**2))
    loss1=float(np.sum(weight[:,None,None]*(np.concatenate([after[b] for b in ('S','O','sum')])-target)**2))
    if loss1>loss0*(1+1e-10)+1e-12:raise ArithmeticError('event substitution increased full-context source error')
    return new_words,dict(before=loss0,after=loss1,changed=totals,limit=maximum)
