"""Freeze temporal subspaces on training people, then compare three paired EEG targets."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml

from src.data.dual_component_tokenizer import (BANDS, base_tokenizer_training_data, base_tokenizer_batch,
    voltage_power_cap_bank, voltage_power_cap_frames, voltage_power_cap_reference)
from src.data.so_supervision_cache import bind_reference, write_json
from src.data.temporal_projection_so import BLOCK, RATE, complete_blocks, project_blocks, rank_frequencies

CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/temporal_projection_so_v1.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
OUT = ROOT/CFG['output']
PARENT = ROOT/CFG['parent']
SSM = ROOT/CFG['ssm']
BAND_NAMES = ('delta','theta','alpha','beta','gamma')


def read(p):
    return json.loads(p.read_text())


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def corr(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(a*b,axis=0)/np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))


def band_bins(x):
    # Discard 1 s at either end before making integer-second envelopes.
    waves=voltage_power_cap_bank(x)[RATE:-RATE,:5]
    return np.mean(waves.reshape(-1,RATE,5,x.shape[1])**2,axis=1)


def describe(s,old,old_power):
    power=band_bins(s)
    ratio=np.median(power.mean(0)/old_power.mean(0),axis=1)
    envelope=np.median(corr(power,old_power),axis=1)
    direction=np.median(np.mean(np.sign(np.diff(power,axis=0))==np.sign(np.diff(old_power,axis=0)),axis=0),axis=1)
    sel=slice(RATE,-RATE)
    row=dict(s_energy=float(np.median(np.mean(s[sel]**2,axis=0)/np.mean(old[sel]**2,axis=0))))
    for j,b in enumerate(BAND_NAMES):
        row.update({b+'_power':ratio[j],b+'_env_r':envelope[j],b+'_direction':direction[j]})
    if not all(np.isfinite(list(row.values()))):
        raise ValueError('Undefined physiological attribute')
    return row


def semantics_ok(table):
    by_ds=table.groupby('dataset').median(numeric_only=True)
    return all((by_ds[k.removesuffix('_min')]>=v).all() for k,v in CFG['semantics'].items())


def train_panel():
    cfg=yaml.safe_load((ROOT/CFG['training_reader']).read_text())
    m,scales,_,_=base_tokenizer_training_data(cfg,ROOT,recorded_root=CFG['recorded_root'])
    rows=m['rows']
    subjects={}
    for i,r in sorted(enumerate(rows), key=lambda z:z[1]['parent']):
        if r['modality']!='eeg' or r['split']!='train': continue
        key=r['dataset'],r['subject']
        for g in sorted(r['inputs']['score'],key=lambda g:g['start_s']):
            if g['positions'] and key not in subjects:
                subjects[key]=(i,g)
    if len(subjects)!=60: raise ValueError('Training census drift')
    panel=[]
    for ds in sorted({k[0] for k in subjects}):
        for j,key in enumerate(sorted(k for k in subjects if k[0]==ds)):
            i,g=subjects[key]
            r=rows[i]
            batch=base_tokenizer_batch(rows,[(i,g['start_s'],p) for p in g['positions']],scales,array_cache=OrderedDict())
            s=batch['s'].reshape(len(g['positions']),-1).T.astype(float)
            mask=batch['mask'].reshape(s.shape[1],-1).T
            origin=round(r['native_start_sample']*200/r['native_rate_hz'])+g['start_s']*200
            blocks,sl=complete_blocks(s,origin)
            mblocks=mask[sl].reshape(blocks.shape)
            keep=mblocks.all(axis=(1,2))
            indices=np.flatnonzero(keep)
            if len(indices)<10 or np.any(np.diff(indices)!=1):raise ValueError('Incomplete calibration support')
            panel.append((dict(dataset=ds,subject=key[1],role=('rank' if j%2==0 else 'select'),
                               parent=r['parent'],crop_start_s=g['start_s'],positions=g['positions'],
                               absolute_start_200=int(origin+sl.start+indices[0]*BLOCK),blocks=len(indices)),
                          blocks[indices]))
    return panel


def calibrate():
    panel=train_panel()
    spectra=[]
    for m,b in panel:
        if m['role']=='rank':
            en=(abs(np.fft.rfft(b,axis=1))**2).mean(0)
            spectra.append((m['dataset'],np.mean(en/en[2:90].sum(axis=0),axis=1)))
    mean=np.mean([np.mean([v for d,v in spectra if d==ds],axis=0) for ds in sorted(set(d for d,v in spectra))],axis=0)
    order=rank_frequencies(mean)
    rows=[]
    for m,b in panel:
        if m['role']!='select':continue
        old=b.reshape(-1,b.shape[-1])
        power=band_bins(old)
        for n in CFG['pair_counts']:
            rows.append(dict(dataset=m['dataset'],subject=m['subject'],pairs=n,
                             **describe(project_blocks(b,order[:n]),old,power)))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'training_rank_curve.csv',index=False)
    eligible=[n for n in CFG['pair_counts'] if n<=CFG['candidate_max_pairs'] and semantics_ok(table[table.pairs==n])]
    result=dict(panel=[m for m,_ in panel],order=order,pairs=(min(eligible) if eligible else CFG['candidate_max_pairs']),
                eligible=eligible,training_gate=bool(eligible),config=CFG)
    write_json(OUT/'frozen.json',result)
    return result


def physics(s,model,proxy,positions,first):
    """Largest generalized eigenvalue of final six-band spatial second moments."""
    frames,clock=voltage_power_cap_frames(voltage_power_cap_bank(s))
    use=(clock>=1)&(clock<=len(s)/RATE-1)
    ref=voltage_power_cap_reference(model,proxy,clock[use]+first/RATE)[:,:,positions]
    frames=frames[use]
    spatial=model['correlation'][:,positions][:,:,positions]
    inv=np.linalg.inv(np.linalg.cholesky(spatial))
    values=[]
    for k in range(0,len(frames),12):
        z=inv @ (frames[k:k+12]/np.sqrt(ref[k:k+12])[...,None])
        values.append(np.linalg.eigvalsh(z@z.swapaxes(-1,-2)/RATE)[...,-1])
    return np.concatenate(values)


def assess(item):
    c,frozen=item
    i=c['i']; indices=c['valid']; src=c['source']
    if src['split']!='validation' or src['rate']!=200:raise ValueError('Wrong validation coordinates')
    base_id=dict(i=i,dataset=c['dataset'],subject=c['subject'])
    order=frozen['order']; n=frozen['pairs']
    folder=OUT/'targets'/str(i);folder.mkdir(exist_ok=True,parents=True)
    with np.load(SSM/'references'/(c['dataset']+'.npz')) as f:
        model=bind_reference(dict(f),np.load(SSM/'inputs'/str(i)/'projection.npy'),900)
    baseline={}; rows=[]; physics_rows=[]; intervention=[]; curve=[]
    for v in CFG['variants']:
        with np.load(PARENT/'targets'/str(i)/(v+'.npz')) as f:
            np.testing.assert_array_equal(f['original_channel_index'],indices)
            x=f['x']; old=f['v3']; added=f['delta']
        # The final 0.1 s is outside source-scoring support.
        b,sl=complete_blocks(old[:-20],src['start_sample'])
        xb,_=complete_blocks(x[:-20],src['start_sample'])
        x=x[sl];old=old[sl];added=added[sl]
        old_power=band_bins(old)
        arms=dict(v3=old,projected_v3=project_blocks(b,order[:n]),projected_x=project_blocks(xb,order[:n]))
        score=slice(RATE,-RATE)
        with np.load(SSM/'fits'/str(i)/'unified'/(v+'.npz')) as f:
            proxy=f['prediction'][:,:,0]
        save=dict(start_in_window=sl.start,stop_in_window=sl.stop,absolute_start_200=src['start_sample']+sl.start,
                  original_channel_index=indices)
        if v=='baseline':
            save.update(x=x, v3=old)
            for k in CFG['pair_counts']:
                candidate=arms['projected_v3'] if k==n else project_blocks(b,order[:k])
                curve.append(dict(**base_id,pairs=k,**describe(candidate,old,old_power)))
        for a,s in arms.items():
            o=x-s
            closure=np.max(np.abs(s+o-x))/rms(x)
            assert closure<1e-10
            if a!='v3':
                save[a]=s
                if v=='baseline':save[a+'_o']=o
            if v in ('baseline','ocular1'):
                ratios=physics(s,model,proxy,indices,sl.start)
                save[a+'_cap_ratio']=ratios
                physics_rows.append(dict(**base_id,arm=a,variant=v,p90=np.quantile(ratios,.9),
                    median=np.median(ratios),maximum=ratios.max(),above_one=np.mean(ratios>1+1e-8),
                    frames=ratios.shape[0],bands=ratios.shape[1]))
            if v=='baseline':
                baseline[a]=s
                jumps=np.arange(BLOCK,len(s),BLOCK)
                rows.append(dict(**base_id,arm=a,closure=closure,**describe(s,old,old_power),
                    s_over_x=rms(s[score])/rms(x[score]),o_over_x=rms(o[score])/rms(x[score]),
                    o_over_old_o=rms(o[score])/rms((x-old)[score]),sx_r=np.median(corr(s[score],x[score])),
                    boundary_jump_ratio=rms(s[jumps]-s[jumps-1])/rms(np.diff(s,axis=0))))
            else:
                ds=s-baseline[a]
                active=slice(6000-sl.start,10000-sl.start)
                d=added[active];ds_a=ds[active];do=d-ds_a
                intervention.append(dict(**base_id,arm=a,variant=v,closure=closure,leak=rms(ds_a)/rms(d),
                    s_relative=rms(ds_a)/rms(baseline[a][active]),s_pair_corr=corr(s[active].ravel(),baseline[a][active].ravel()),
                    o_corr=corr(do.ravel(),d.ravel()),s_gain=np.sum(ds_a*d)/np.sum(d*d),
                    full_change_rms=rms(ds)))
        if v!='baseline':
            assert rms(arms['projected_v3']-baseline['projected_v3'])<=rms(old-baseline['v3'])*(1+1e-10)
        temp=folder/(v+'.partial.npz');np.savez(temp,**save);temp.replace(folder/(v+'.npz'))
    return rows,physics_rows,intervention,curve


def natural_eog(cases):
    from experiments.scripts.score_soft_power_so import eog_basis
    from src.data.downstream_input_cache import DownstreamInputDataset
    from experiments.scripts.evaluate_unified_eeg_so_pilot import CFG as prior
    view=DownstreamInputDataset(ROOT/prior['downstream'],split='validation')
    rows=[]
    for c in cases:
        q=eog_basis(view,c)
        if q is None:continue
        with np.load(OUT/'targets'/str(c['i'])/'baseline.npz') as f:
            # Restrict and re-orthonormalize the EOG space on this exact scoring interval.
            first=int(f['start_in_window'])+RATE;last=int(f['stop_in_window'])-RATE
            z=q[first:last]; q=np.linalg.qr(z-z.mean(0))[0]
            front=[j for j,idx in enumerate(c['valid']) if c['channels'][idx].upper().startswith(('FP','AF'))]
            def projection(key):
                a=f[key][RATE:-RATE,front];return np.linalg.norm(q.T@(a-a.mean(0)))
            for arm in CFG['arms']:
                rows.append(dict(i=c['i'],dataset=c['dataset'],subject=c['subject'],arm=arm,
                                 eog_ratio=projection(arm)/projection('x'),frontal_sites=len(front)))
    return pd.DataFrame(rows)


def run(workers):
    if (OUT/'summary.json').exists():raise FileExistsError('Completed run is immutable')
    if read(PARENT/'summary.json')['status']!='completed':raise ValueError('Incomplete parent pilot')
    start=time.monotonic()
    frozen=calibrate()
    cases=read(PARENT/'plan.json')['cases']
    assert len(cases)==len({(c['dataset'],c['subject']) for c in cases})==8
    assert not {(c['dataset'],c['subject']) for c in cases}&{(t['dataset'],t['subject']) for t in frozen['panel']}
    write_json(OUT/'plan.json',dict(config=CFG,cases=cases,workers=workers))
    print('frozen',frozen['pairs'],frozen['training_gate'],round(time.monotonic()-start,1),flush=True)
    with ProcessPoolExecutor(max_workers=min(workers,len(cases))) as pool:
        parts=list(pool.map(assess,[(c,frozen) for c in cases]))
    tables=[]
    for j,name in enumerate(['baseline','final_spatial_cap','interventions','validation_rank_curve']):
        df=pd.DataFrame([row for part in parts for row in part[j]])
        df.to_csv(OUT/(name+'.csv'),index=False);tables.append(df)
    base,cap,change,_=tables
    eog=natural_eog(cases);eog.to_csv(OUT/'natural_eog.csv',index=False)
    ocular=change.query("variant=='ocular1'")
    bounds=CFG['validation_gate']
    checks=dict(training_semantics=frozen['training_gate'],validation_semantics=semantics_ok(base[base.arm=='projected_v3']))
    effects=[]
    for frame,field,name,action in [(ocular,'leak','leak_increase','diff'),(ocular,'s_relative','relative_s_change_increase','diff'),
        (eog,'eog_ratio','natural_eog_increase','diff'),(cap[cap.variant=='baseline'],'p90','baseline_cap_p90_ratio','ratio'),
        (cap[cap.variant=='ocular1'],'p90','ocular_cap_p90_ratio','ratio')]:
        wide=frame.pivot(index='i',columns='arm',values=field)
        v=wide['projected_v3']/wide['v3'] if action=='ratio' else wide['projected_v3']-wide['v3']
        checks[name]=bool(v.median()<=bounds['median_'+name+'_max'])
        effects.append(dict(metric=name,median=float(v.median()),threshold=bounds['median_'+name+'_max'],n=len(v)))
    wide=ocular.pivot(index='i',columns='arm',values='leak')
    checks['less_leak_than_px']=bool((wide.projected_v3-wide.projected_x).median()<0)
    pd.DataFrame(effects).to_csv(OUT/'paired_gates.csv',index=False)
    write_json(OUT/'summary.json',dict(status='completed',subjects=len(cases),electrode_instances=sum(len(c['valid']) for c in cases),
        planned_cells=96,completed_cells=len(base)+len(change),pairs=frozen['pairs'],real_dof=2*frozen['pairs'],
        checks=checks,advance_to_training=all(checks.values()),seconds=time.monotonic()-start,
        max_closure_over_x_rms=float(max(base.closure.max(),change.closure.max()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.workers)
