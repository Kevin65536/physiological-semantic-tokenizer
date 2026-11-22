"""Input/reference crossover for the retained eight-window unified EEG ocular pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import time

import evaluate_unified_eeg_so_pilot as pilot
import numpy as np
import pandas as pd
from scipy.signal import welch
from src.data.dual_component_tokenizer import (project_voltage_second_moment, voltage_power_cap_bank,
    voltage_power_cap_frames, voltage_power_cap_reference)

RUN = pilot.OUT
DEFAULT = RUN/'visual_ocular_mechanism_v1'
COMMON = ['FP1','FP2','C3','C4','CZ','O1','O2','P3','P4','P7','P8','T7','T8']
SCORE = slice(6000,10000)
# Only frames with nonzero overlap with [30,50) are needed. These indices are on the padded frame grid.
FRAMES = np.arange(296,505)
RMS = pilot.rms


def ocular_weights(names):
    def weight(k):
        if k.startswith(('FP','AFP')):return 1.
        if k.startswith('AF'):return .65
        if k.startswith('FC'):return .22
        if k.startswith('F'):return .4
        return .08
    return np.array([weight(k) for k in names])


def frames(x):
    bank=voltage_power_cap_bank(x,sample_rate_hz=200)
    view,_=voltage_power_cap_frames(bank)
    return view[FRAMES[0]:FRAMES[-1]+1]


def projected_front(frame_view, cov, front=0):
    """Use the production projection and Hann sum; return the scored electrode only."""
    shape=(len(FRAMES)-1)*20+200
    acc=np.zeros(shape);weights=np.zeros(shape)
    hann=np.hanning(201)[:-1]
    for s in range(0,len(FRAMES),8):
        block,_=project_voltage_second_moment(frame_view[s:s+8],cov[s:s+8],1.)
        for k,wave in enumerate(block[:,:,front].sum(1)):
            start=(s+k)*20
            acc[start:start+200]+=wave*hann
            weights[start:start+200]+=hann
    offset=SCORE.start+100-FRAMES[0]*20
    use=slice(offset,offset+4000)
    return acc[use]/weights[use]


def pack(i):
    ident,x1,rate,d,*_=pilot.known_inputs(i,'unified','ocular1')
    g=ident['group'];names=[n.upper() for n in g['EEG_channels']]
    assert rate==200 and g['split']=='validation'
    wd=RUN/'inputs'/str(i)
    x=np.load(wd/'unified.npy',mmap_mode='r')
    j=next(k for k,n in enumerate(names) if n.startswith(('FP','AFP')))
    s0=np.load(RUN/'targets'/str(i)/'unified/baseline_S.npy',mmap_mode='r')
    coeff=ocular_weights(names)
    np.testing.assert_allclose(d,d[:,j,None]*coeff,rtol=1e-8,atol=max(abs(d).max()*1e-12,1e-20))
    with np.load(RUN/'fits'/str(i)/'unified/baseline.npz') as a:pred=a['prediction'][:,:,0]
    with np.load(RUN/'references'/(g['dataset']+'.npz')) as a:
        model=pilot.bind_reference(dict(a),np.load(wd/'projection.npy'),900)
    power=voltage_power_cap_reference(model,pred,FRAMES/10)
    root=np.sqrt(power)
    cov=root[:,:,:,None]*root[:,:,None,:]*model['correlation']
    return dict(i=i,dataset=g['dataset'],subject=g['subject'],names=names,j=j,x=x,d=d,
                s=s0,scale=RMS(x[SCORE,j]),w=coeff,cov=cov)


def descriptive(x,names,j,w,rate):
    # Center per channel for descriptive spatial covariance only (the production projector does not center).
    a=x-x.mean(0); q=a[:,j]
    t=names.index('CZ')
    fp2=names.index('FP2') if 'FP2' in names else names.index('AFP2')
    f,p=welch(a,fs=rate,nperseg=4*rate,axis=0)
    total=p[:,j].sum();pmax=f[np.argmax(p[:,j])]
    align=np.sum((a@w)**2)/(w@w*np.sum(a*a))
    corr=np.corrcoef(a.T)
    return dict(front_delta_share=p[(f>=1)&(f<4),j].sum()/total,
                front_alpha_share=p[(f>=8)&(f<13),j].sum()/total,
                front_peak_hz=pmax,front_cz_r=corr[j,t],front_other_frontal_r=corr[j,fp2],
                spatial_template_fraction=align,
                front_to_median_channel_rms=RMS(q)/np.median(np.std(a,axis=0)),
                spatial_mean_fraction=np.mean(np.mean(a,axis=1)**2)/np.mean(a*a))


def features_one(i):
    a=CASES[i];j=a['j'];w=a['w'];cov=a['cov']
    x=a['x'];native=np.load(RUN/'inputs'/str(i)/'native.npy',mmap_mode='r')
    rate=len(native)//90
    meta=dict(i=i,subject=a['subject'],dataset=a['dataset'])
    result=[]
    for stage,data,fs in [('native',native,rate),('unified',x,200)]:
        result.append(dict(meta,stage=stage,**descriptive(data[30*fs:50*fs],a['names'],j,w,fs)))
    # Ellipsoid radius for a pure pulse with the injection map w (w_front = 1).
    radius=1/np.sqrt(np.einsum('c,tbc->tb',w,np.linalg.solve(cov,np.broadcast_to(w,(*cov.shape[:-1],))[...,None])[...,0]))
    band=voltage_power_cap_bank(x)
    db=voltage_power_cap_bank(a['d'][:,j:j+1])
    inj_band=np.mean(db[SCORE,:,0]**2,axis=0)
    fx=frames(x); fd=frames(a['d'])
    original=projected_front(fx,cov,j)
    np.testing.assert_allclose(original,a['s'][SCORE,j],rtol=1e-9,atol=a['scale']*1e-10)
    leak=RMS(projected_front(fx+fd,cov,j)-original)/RMS(a['d'][SCORE,j])
    previous=HIST.iloc[i]
    np.testing.assert_allclose(leak,previous.front_fixed_leak,atol=1e-8)
    mechanism=dict(meta,front_rms_over_dataset_training= a['scale']/REF_RMS[a['dataset']][j],
                   modeled_delta_direction_radius=np.median(radius[:,0])/a['scale'],
                   modeled_delta_front_sd=np.median(np.sqrt(cov[:,0,j,j]))/a['scale'],
                   delta_template_fraction=descriptive(band[SCORE,0],a['names'],j,w,200)['spatial_template_fraction'],
                   input_delta_front_rms=RMS(band[SCORE,0,j])/a['scale'],
                   injection_delta_carrier_share=inj_band[0]/inj_band.sum(),replay_leak=leak)
    return result,mechanism


def common_inputs(x,scale,sign=False):
    raw=x/scale
    weights=ocular_weights(COMMON)
    # Independent pre-injection support chooses a polarity map; energy at each electrode stays identical.
    polarity=np.sign((raw[1000:5000]-raw[1000:5000].mean(0)).T@(raw[1000:5000,0]-raw[1000:5000,0].mean()))
    polarity[polarity==0]=1.
    w=weights*polarity if sign else weights
    injected=TEMPLATE[:,None]*w
    return frames(raw),frames(injected),int(np.sum(polarity<0))


def cross_one(i):
    a=CASES[i]
    indices=[a['names'].index(n) for n in COMMON]
    raw=a['x'][:,indices];fx,fd,flips=common_inputs(raw,a['scale'])
    _,fd_sign,_=common_inputs(raw,a['scale'],True)
    rows=[]
    for k in SELECTED:
        ref=COVS[k]
        base=projected_front(fx,ref)
        for dose in (1.,2.,4.):
            leak=RMS(projected_front(fx+dose*fd,ref)-base)/dose
            rows.append(dict(input=i,reference=k,dose=dose,variant='positive',leak=leak))
        if k==i:
            rows.append(dict(input=i,reference=k,dose=2.,variant='prewindow_polarity',flips=flips,
                             leak=RMS(projected_front(fx+2*fd_sign,ref)-base)/2))
    return rows


def extra_plan():
    groups=pilot.read_json(pilot.SO/'plan.json')['groups']
    used={a['key'] for a in [pilot.read_json(RUN/'inputs'/str(i)/'identity.json')['group'] for i in SELECTED]}
    old=[g for g in groups if g['key'] in used];avail={}
    for g in groups:
        if g['split']!='validation' or g['duration_s']!=90 or not set(COMMON)<=set(n.upper() for n in g['EEG_channels']):continue
        if any(g['dataset']==h['dataset'] and g['subject']==h['subject'] and g['native_record']==h['native_record']
               and abs(g['start_sample']-h['start_sample'])<90*g['rate'] for h in old):continue
        avail.setdefault((g['dataset'],g['subject']),[]).append(g)
    selected=[]
    for _,options in sorted(avail.items()):
        seq=sorted(options,key=lambda g:(g['native_record'],g['start_sample']))
        first=seq[0]; selected.append(first)
        other=[g for g in seq if g['native_record']!=first['native_record'] or
               abs(g['start_sample']-first['start_sample'])>=90*g['rate']]
        if other:selected.append(other[-1])
    return selected


def extra_one(g):
    names=[n.upper() for n in g['EEG_channels']]
    x=pilot.unified(g)[:,[names.index(n) for n in COMMON]]
    scale=RMS(x[SCORE,0]);fx,fd,_=common_inputs(x,scale)
    base=projected_front(fx,POOLED)
    return dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],record=g['native_record'],
                original_subject=g['subject'] in {CASES[i]['subject'] for i in SELECTED},
                leak=RMS(projected_front(fx+2*fd,POOLED)-base)/2,
                **descriptive(x[SCORE],COMMON,0,ocular_weights(COMMON),200))


def run(fn,items,workers):
    begin=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork')) as pool:
        futures={pool.submit(fn,it):it for it in items}
        for n,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if n==1 or n==len(items):print(fn.__name__,n,len(items),round(time.monotonic()-begin,2),flush=True)
    return results


def fixture():
    rng=np.random.default_rng(64)
    x=rng.normal(size=(18000,2))
    fx=frames(x)
    cov=np.broadcast_to(np.eye(2)*100,(*fx.shape[:2],2,2))
    np.testing.assert_allclose(projected_front(fx,cov),x[SCORE,0],rtol=1e-10,atol=1e-10)
    v=np.array([.7,-1.3]);t=np.sin(np.arange(18000)/20)
    a=frames(t[:,None]*v)
    np.testing.assert_allclose(projected_front(a,cov),t[SCORE]*v[0],atol=1e-12)


def main(out,workers):
    global CASES,HIST,REF_RMS,TEMPLATE,SELECTED,COVS,POOLED
    fixture()
    assert pilot.read_json(RUN/'verification.json')['status']=='passed'
    out.mkdir(parents=True,exist_ok=False)
    HIST=pd.read_csv(RUN/'ocular_visual_comparison_v1/ocular_visual_comparison.csv')
    CASES=[pack(i) for i in range(8)]
    assert list(HIST.subject.astype(str))==[str(a['subject']) for a in CASES]
    REF_RMS={}
    for d in {a['dataset'] for a in CASES}:
        with np.load(RUN/'references'/(d+'.npz')) as f:REF_RMS[d]=f['voltage_rms']
    SELECTED=[a['i'] for a in CASES if set(COMMON)<=set(a['names'])]
    assert SELECTED==[2,3,4,5,6,7]
    TEMPLATE=CASES[2]['d'][:,CASES[2]['j']].copy();TEMPLATE/=RMS(TEMPLATE[SCORE])
    COVS={}
    for i in SELECTED:
        a=CASES[i];ix=[a['names'].index(n) for n in COMMON]
        COVS[i]=a['cov'][:,:,ix][:,:,:,ix]/a['scale']**2
    POOLED=np.mean(list(COVS.values()),axis=0)
    extras=extra_plan()
    pilot.write_json(out/'design.json',dict(inputs='retained public validation only; no new model fits',
        cases=[str(a['subject']) for a in CASES],common=COMMON,crossed=SELECTED,extrawindows=[g['key'] for g in extras],
        rules='Six baseline frozen reference covariance trajectories, divided by squared donor pre-injection frontal RMS; each 13-channel input divided by its own frontal RMS in 30–50s. Identical 200 Hz donor waveform and topology; doses 1, 2 and 4 in frontal input RMS, not baseline S. Main comparison dose 2. Other windows take first and last eligible disjoint 90 s by record/time, excluding original physical supports. No outcomes used for window selection.'))
    results=run(features_one,range(8),workers)
    pd.DataFrame([m for rows,_ in results for m in rows]).sort_values(['i','stage']).to_csv(out/'features.csv',index=False)
    pd.DataFrame([m for _,m in results]).sort_values('i').to_csv(out/'native_montage.csv',index=False)
    results=run(cross_one,SELECTED,workers)
    pd.DataFrame([m for rows in results for m in rows]).sort_values(['dose','input','reference','variant']).to_csv(out/'crossover.csv',index=False)
    pilot.INDEX,pilot.ENTRIES=pilot.downstream()
    ext=pd.DataFrame(run(extra_one,extras,workers)).sort_values(['dataset','subject','record'])
    ext.to_csv(out/'extra_windows.csv',index=False)
    ext.groupby(['dataset','subject']).mean(numeric_only=True).to_csv(out/'extra_subjects.csv')
    pilot.write_json(out/'verification.json',dict(status='passed',n_source_replays=8,synthetic_overlap_add='passed',
        reference_replay_max_abs=float(np.max(abs(pd.read_csv(out/'native_montage.csv').replay_leak.values-HIST.front_fixed_leak.values))),
        complete_crosses=3*len(SELECTED)**2,polarity_controls=len(SELECTED),extra_windows=len(ext),
        extra_subjects=len(ext.groupby(['dataset','subject']))))


def subset_cov(a,names):
    ix=[a['names'].index(n) for n in names]
    return a['cov'][:,:,ix][:,:,:,ix]/a['scale']**2


def target_control(a,names,ref):
    ix=[a['names'].index(n) for n in names]
    x=a['x'][:,ix]/a['scale']
    fx=frames(x)
    fd=frames(TEMPLATE[:,None]*ocular_weights(names))
    j=names.index(a['names'][a['j']])
    base=projected_front(fx,ref,j)
    return RMS(projected_front(fx+2*fd,ref,j)-base)/2


def montage_one(i):
    a=CASES[i];names=a['names']
    layouts={'full':names}
    if set(COMMON)<=set(names):layouts['common13']=COMMON
    if set(VIS)<=set(names):
        extra=[c for c in VIS if c not in COMMON]
        frontal=[c for c in extra if c.startswith(('AF','F')) and not c.startswith('FC')]
        layouts.update(common31=VIS, add_frontal=COMMON+frontal,
                       add_other=COMMON+[c for c in extra if c not in frontal])
    rows=[]
    for name,chs in layouts.items():
        if name=='common31' and chs==names and rows:
            value=rows[0]['leak']
        else:value=target_control(a,chs,subset_cov(a,chs))
        rows.append(dict(i=i,subject=a['subject'],layout=name,n_channels=len(chs),leak=value,channels=','.join(chs)))
    return rows


def fine_one(i):
    a=CASES[i];ix=[a['names'].index(n) for n in VIS]
    x=a['x'][:,ix]/a['scale']
    fx=frames(x)
    fd=frames(TEMPLATE[:,None]*ocular_weights(VIS))
    rows=[]
    for k,R in FINE_COV.items():
        sqrt=np.sqrt(np.diagonal(R,axis1=-2,axis2=-1))
        for corr_source in (4,6):
            reference=FINE_COV[corr_source]
            margins=np.sqrt(np.diagonal(reference,axis1=-2,axis2=-1))
            correlation=reference/(margins[...,None]*margins[...,None,:])
            cov=sqrt[...,None]*correlation*sqrt[...,None,:]
            if (k<6)==(corr_source<6):np.testing.assert_allclose(cov,R,atol=1e-12,rtol=1e-11)
            base=projected_front(fx,cov)
            leak=RMS(projected_front(fx+2*fd,cov)-base)/2
            rows.append(dict(input=i,power=k,spatial=corr_source,leak=leak))
    return rows


def other31(g):
    names=[n.upper() for n in g['EEG_channels']];ix=[names.index(n) for n in VIS]
    x=pilot.unified(g)[:,ix];scale=RMS(x[SCORE,0]);fx=frames(x/scale)
    fd=frames(TEMPLATE[:,None]*ocular_weights(VIS))
    blocks=[]
    for k in (4,5,6,7):
        base=projected_front(fx,FINE_COV[k])
        leak=RMS(projected_front(fx+2*fd,FINE_COV[k])-base)/2
        blocks.append(dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],reference=k,leak=leak))
    return blocks


def spatial(out,workers):
    global CASES,TEMPLATE,FINE_COV,VIS,SELECTED
    fixture()
    require_success=pilot.read_json(out/'verification.json')['status']=='passed'
    assert require_success
    CASES=[pack(i) for i in range(8)];VIS=CASES[6]['names']
    assert len(VIS)==31 and VIS==CASES[7]['names'] and VIS[0]=='FP1'
    SELECTED=[2,3,4,5,6,7]
    TEMPLATE=CASES[2]['d'][:,CASES[2]['j']].copy();TEMPLATE/=RMS(TEMPLATE[SCORE])
    FINE_COV={i:subset_cov(CASES[i],VIS) for i in (4,5,6,7)}
    if (out/'montage.csv').exists():
        levels=pd.read_csv(out/'montage.csv').to_dict('records')
    else:
        levels=[r for result in run(montage_one,range(8),workers) for r in result]
        pd.DataFrame(levels).sort_values(['i','layout']).to_csv(out/'montage.csv',index=False)
    if (out/'crossover31.csv').exists():
        fine=pd.read_csv(out/'crossover31.csv')
    else:
        crossed=[r for result in run(fine_one,list(FINE_COV),workers) for r in result]
        fine=pd.DataFrame(crossed).sort_values(['input','power','spatial'])
        fine.to_csv(out/'crossover31.csv',index=False)
    assert len(fine)==32 and not fine.duplicated(['input','power','spatial']).any()
    for i in FINE_COV:
        own=fine[(fine.input==i)&(fine.power==i)&(fine.spatial==(4 if i<6 else 6))].leak.item()
        compare=next(r['leak'] for r in levels if r['i']==i and r['layout']=='common31')
        np.testing.assert_allclose(own,compare,atol=1e-11)
    pilot.INDEX,pilot.ENTRIES=pilot.downstream()
    extra=[g for g in extra_plan() if set(VIS)<=set(n.upper() for n in g['EEG_channels'])]
    ext=pd.DataFrame([r for batch in run(other31,extra,workers) for r in batch])
    ext.sort_values(['key','reference']).to_csv(out/'extra31.csv',index=False)
    pilot.write_json(out/'spatial_verification.json',dict(status='passed',crossed=len(fine),extra_windows=len(extra),
        extra_subjects=len(ext.groupby(['dataset','subject'])),own_diagonals_verified=4,notes='FP1 RMS matched, common donor, dose 2; power and spatial reference exchanged independently.'))



def reference_source_one(i):
    a=CASES[i];wd=RUN/'inputs'/str(i);ix=[a['names'].index(n) for n in VIS]
    x=a['x'][:,ix]/a['scale'];fx=frames(x)
    w=ocular_weights(VIS)
    fd=frames(TEMPLATE[:,None]*w)
    # Identical frequency control at Fp1 alone, same 20 s envelope and Fp1 dose.
    t=np.arange(18000)/200
    carrier=np.sin(2*np.pi*10*t)*np.where((t>=30)&(t<50),np.sin(np.pi*(t-30)/20)**2,0.)
    carrier/=RMS(carrier[SCORE])
    focal=np.zeros(len(VIS));focal[0]=1.
    fn=frames(carrier[:,None]*focal)
    rows=[]
    for k in FINE_COV:
        b=CASES[k];sel=[b['names'].index(n) for n in VIS]
        with np.load(RUN/'references'/(b['dataset']+'.npz')) as f:
            m=pilot.bind_reference(dict(f),np.load(RUN/'inputs'/str(k)/'projection.npy'),900)
        for mode in ('baseline_proxy','training_constant'):
            if mode=='baseline_proxy':cov=FINE_COV[k]
            else:
                power=voltage_power_cap_reference(m,None,FRAMES/10)[:,:,sel]/b['scale']**2
                mag=np.sqrt(power);cov=mag[:,:,:,None]*mag[:,:,None,:]*m['correlation'][:,sel][:,:,sel]
            base=projected_front(fx,cov)
            leak=RMS(projected_front(fx+2*fd,cov)-base)/2
            err=RMS(projected_front(fx+2*fn,cov)-base-2*carrier[SCORE])/2
            rows.append(dict(input=i,reference=k,mode=mode,leak=leak,base_front_s_rms_over_x=RMS(base),
                             alpha_focal_preservation_error=err,
                             delta_front_ref_sd=np.median(np.sqrt(cov[:,0,0,0])),
                             delta_fp1_cz_reference_correlation=m['correlation'][0,sel[0],sel[VIS.index('CZ')]]))
    return rows


def reference_source(out,workers):
    global CASES,TEMPLATE,FINE_COV,VIS
    assert pilot.read_json(out/'spatial_verification.json')['status']=='passed'
    CASES=[pack(i) for i in range(8)];VIS=CASES[6]['names']
    TEMPLATE=CASES[2]['d'][:,CASES[2]['j']].copy();TEMPLATE/=RMS(TEMPLATE[SCORE])
    FINE_COV={i:subset_cov(CASES[i],VIS) for i in (4,5,6,7)}
    rows=[r for group in run(reference_source_one,list(FINE_COV),workers) for r in group]
    df=pd.DataFrame(rows).sort_values(['input','reference','mode'])
    df.to_csv(out/'reference_source.csv',index=False)
    fine=pd.read_csv(out/'crossover31.csv')
    for r in df[df['mode']=='baseline_proxy'].itertuples():
        old=fine[(fine.input==r.input)&(fine.power==r.reference)&(fine.spatial==(4 if r.reference<6 else 6))].leak.item()
        np.testing.assert_allclose(r.leak,old,atol=1e-11)



def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert pilot.read_json(out/'spatial_verification.json')['status']=='passed'
    previous=pd.read_csv(RUN/'ocular_visual_comparison_v1/ocular_visual_comparison.csv')
    labels={i:str(previous.iloc[i].subject) for i in range(8)}
    fine=pd.read_csv(out/'crossover31.csv')
    keep=fine[(fine.power<6)==(fine.spatial<6)]
    matrix=keep.pivot(index='input',columns='power',values='leak')
    components=fine.assign(bg=np.where(fine.input<6,'REFED','Visual'),
                            pow=np.where(fine.power<6,'R','V'),coupling=np.where(fine.spatial<6,'R','V'))
    components=components.groupby(['bg','pow','coupling']).leak.mean().unstack([1,2])
    extras=pd.read_csv(out/'extra31.csv')
    extras['donor']=np.where(extras.reference<6,'REFED','Visual')
    extra=extras.groupby(['dataset','subject','donor']).leak.mean().unstack()
    feature=pd.read_csv(out/'features.csv')
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0]
    for i,x in feature.groupby('i'):
        native=x[x.stage=='native'].front_cz_r.item()
        processed=x[x.stage=='unified'].front_cz_r.item()
        ax.plot([i,i],[native,processed],color='#9caab2')
        ax.plot(i,native,'o',mfc='white',mec='#415a74',label='发布输入' if i==0 else None)
        ax.plot(i,processed,'s',color='#ad4c32',label='统一输入' if i==0 else None)
    ax.set(title='输入特征：额前与 Cz 电压波动方向',ylabel='30–50 秒 Pearson 相关',ylim=(-1.02,1.02),
           xticks=range(8),xticklabels=[labels[i] for i in range(8)])
    ax.tick_params(axis='x',rotation=25);ax.axhline(0,c='#8d959b',lw=.8);ax.legend(frameon=False)
    ax=axes[0,1];im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=.22,aspect='auto')
    for i in range(4):
        for j in range(4):
            val=matrix.iloc[i,j];ax.text(j,i,f'{100*val:.1f}',ha='center',va='center',color='white' if val>.13 else 'black')
    lst=['REFED 8','REFED 31','Visual S06','Visual S14']
    ax.set(title='同一背景改用另一窗口的完整电压参考',xticks=range(4),xticklabels=lst,
           yticks=range(4),yticklabels=lst,xlabel='冻结参考来源',ylabel='输入波形来源')
    cb=fig.colorbar(im,ax=ax);cb.set_label('Fp1 泄漏比（格内为百分数）')
    ax=axes[1,0]
    for i,b in enumerate(['REFED','Visual']):
        ax.bar(np.arange(4)+(i-.5)*.38,[100*components.loc[b,(a,c)] for a,c in [('R','R'),('R','V'),('V','R'),('V','V')]],
               width=.37,color=['#537f96','#a64b33'][i],label=b+' 背景')
    ax.set(title='分别交换频带功率上限和电极相关结构',ylabel='Fp1 泄漏（%）',
           xticks=range(4),xticklabels=['R / R','R / V','V / R','V / V'],xlabel='功率来源 / 相关结构来源（R 为 REFED，V 为 Visual）');ax.legend(frameon=False)
    ax=axes[1,1]
    for (dataset,sub),row in extra.iterrows():
        ax.plot([0,1],100*row[['REFED','Visual']],marker='o' if dataset=='refed' else 's',
                color='#537f96' if dataset=='refed' else '#a64b33',alpha=.8,lw=1.4)
    ax.set(title='额外 18 窗：9 名被试均随参考改变',xticks=[0,1],xticklabels=['REFED 参考','Visual 参考'],
           ylabel='Fp1 泄漏（%，每条线为一人两窗均值）',xlim=(-.20,1.2))
    for ax in (axes[0,0],axes[1,0],axes[1,1]):ax.spines[['top','right']].set_visible(False)
    fig.suptitle('同一眼电供体、31 个匹配电极与冻结预测的对照（输入特征图使用原布局）',fontsize=15)
    fig.savefig(out/'mechanism.png',dpi=220);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--spatial',action='store_true')
    parser.add_argument('--reference-source',action='store_true')
    parser.add_argument('--plot',action='store_true')
    opt=parser.parse_args()
    if opt.plot: render(opt.output)
    elif opt.reference_source: reference_source(opt.output,opt.workers)
    elif opt.spatial: spatial(opt.output,opt.workers)
    else: main(opt.output,opt.workers)
