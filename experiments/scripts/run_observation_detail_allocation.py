#!/usr/bin/env python3
"""Matched bit-allocation diagnostic on the exact, already-selected observation panel."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.observation_tokenizer_evaluation import KEYS, hierarchy_weights, reconstruction, paired_contrast
from src.data.observation_tokenizer_panel import build_panel, PanelLimits, G0_ROOT
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer
from src.tokenizers.adaptive_observation_detail import AdaptiveObservationDetail

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/observation_detail_allocation_v1.yaml'
SUITE=ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
PARENT=SUITE/'20261120_development_v3'


def dump(p, value):
    tmp=p.with_suffix('.partial'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(tmp,p)


def read(p):
    return json.loads(p.read_text())


def configuration(path):
    c=yaml.safe_load(path.read_text()); orig=yaml.safe_load(CONFIG.read_text())
    if (c!=orig or c['experiment_id']!='TK-OBS-ALLOCATION-v1' or ROOT/c['parent']!=PARENT
            or 'no_test' not in c['scope'] or c['rates']!={'eeg':[14,256,512,1024], 'hb':[14,32,64]}):
        raise ValueError('exact development-only allocation protocol required')
    return c


def gate(c):
    paths=[Path(__file__), CONFIG, *(ROOT/k for k in ['src/tokenizers/adaptive_observation_detail.py',
        'src/tokenizers/factorized_observation.py','src/analysis/observation_tokenizer_evaluation.py',
        'src/data/observation_tokenizer_panel.py'])]
    return dict(status='passed',schema=c['schema'],files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})


def synthetic():
    rng=np.random.default_rng(4)
    for m,c,t in [('eeg',1,400),('hb',2,20)]:
        x=rng.normal(size=(48,c,t)); z=x[:5]-.2
        codec=FactorizedObservationTokenizer(m,seed=17).fit(x,partition='train')
        for bits in (14,64):
            ada=AdaptiveObservationDetail(codec,bits).fit(x,partition='train')
            for mode in ['conditional','independent']:
                codes=ada.encode(z,mode=mode); packed=ada.pack(codes,mode=mode)
                assert len(packed)==(len(z)*bits+7)//8
                recovered=ada.decode(ada.unpack(packed,len(z),mode=mode),mode=mode)
                np.testing.assert_array_equal(recovered,ada.decode(codes,mode=mode)); assert recovered.shape==z.shape


def replay_panel(parent):
    """Re-check the original selected row identity and literal 15-bit score mask before pairing."""
    cfg=read(parent/'resolved_config.json'); d=cfg['data']
    if ROOT/d['g0']!=G0_ROOT or d['splits']!=['train','validation'] or d['raw_reads'] or d['test_access']!='forbidden':
        raise ValueError('unexpected parent access scope')
    p=build_panel(G0_ROOT,limits=PanelLimits(**d['panel']),seed=d['panel_seed'])
    cols=['modality','split',*KEYS]; ident=lambda d:tuple(str(d[k]) for k in cols)
    saved=pd.read_csv(parent/'panel.csv',dtype=str,keep_default_na=False); lookup={ident(r):r for r in saved.to_dict('records')}
    ordered=sorted(enumerate(p.rows),key=lambda item:(['eeg','hb'].index(item[1].modality), ['train','validation'].index(item[1].split)))
    actual=[tuple(str(getattr(r,k)) for k in cols) for _,r in ordered]
    if actual!=list(lookup) or len(actual)!=len(saved): raise ValueError('parent ordered identities changed')
    blocks={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        samples=pool.map(p.read,(i for i,_ in ordered))
        for key,(_,r),s in zip(actual,ordered,samples,strict=True):
            old=lookup[key]; good=s['patch_scorable']; included=bool(s['available'] and np.any(good))
            if (''.join(map(str,good.astype(int)))!=old['valid_bits'] or str(included)!=old['included']
                    or float(old['scale'])!=r.scale or str(r.native_start_sample)!=old['native_start_sample']):
                raise ValueError('parent mask, scale or time changed')
            if not included: continue
            rows,x,v=blocks.setdefault((r.modality,r.split),([],[],[])); ids=np.flatnonzero(good)
            x.append(s['x'][ids].transpose(0,2,1)); v.append(np.tile(r.vx,(len(ids),1)))
            rows.extend(dict(**{k:getattr(r,k) for k in KEYS},patch=int(j)) for j in ids)
    return {k:(pd.DataFrame(f),np.concatenate(x),np.concatenate(v)) for k,(f,x,v) in blocks.items()},cfg


def run_one(mod,bits,parent,batch,train,parent_bits):
    tick=time.perf_counter(); frame,z,vx=batch
    codec=joblib.load(parent/f'{mod}_codec.joblib'); blind=joblib.load(parent/f'{mod}_blind_control.joblib')
    def score(x,method,b): return reconstruction(frame,z,x,vx,mod).assign(method=method,bits_per_patch=b,modality=mod,split='validation')
    tables=[]; saved=None
    for c,mode,name in [(codec,'conditional','adaptive_pattern'),(codec,'independent','independent_tail'),(blind,'conditional','adaptive_blind')]:
        if name!='independent_tail': a=AdaptiveObservationDetail(c,bits).fit(train,partition='train')
        else: a=saved
        if name=='adaptive_pattern': saved=a
        code=a.encode(z,mode=mode); expected=c.encode(z,mode='prototype')['pattern']
        if not np.array_equal(expected,code['pattern']): raise ValueError('frozen ID assignment changed')
        payload=a.pack(code,mode=mode)
        if len(payload)!=(len(z)*bits+7)//8: raise ValueError('packed budget mismatch')
        y=a.decode(a.unpack(payload,len(z),mode=mode),mode=mode); tables.append(score(y,name,bits))
    # The extended 1024-bit cell was not in the parent. Refit the same global-flat estimator
    # on the SAME draws; no validation-dependent coefficient, gain, or readout selection.
    if bits not in parent_bits:
        q=FactorizedObservationTokenizer(mod,n_patterns=16,detail_count=(bits-4)//4,detail_bits=4,seed=17).fit(train,partition='train')
        if q.rate('flat')['bits_per_item']!=bits: raise ValueError('expanded flat budget mismatch')
        tables.append(score(q.decode(q.encode(z,mode='flat'),mode='flat'),'flat',bits))
    return bits,saved,pd.concat(tables,ignore_index=True),time.perf_counter()-tick


def screen(c,out):
    parent=PARENT; summary=read(parent/'summary.json'); source=read(parent/'resolved_config.json')
    if (summary['status']!='completed' or summary['experiment_id']!='TK-OBS-FACTOR-v1'
            or not (parent/'source_manifest.json').is_file()): raise ValueError('completed exact parent required')
    for f in ['src/tokenizers/factorized_observation.py','src/data/observation_tokenizer_panel.py','src/analysis/observation_tokenizer_evaluation.py']:
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=summary['sources'][f]: raise ValueError('parent computation drift')
    data, cfg=replay_panel(parent); baseline=pd.read_csv(parent/'reconstruction.csv',dtype={'subject':str},keep_default_na=False)
    baseline['bits_per_patch']=pd.to_numeric(baseline.bits_per_patch,errors='coerce'); allrows=[]; details={}
    for mod,rates in c['rates'].items():
        print('allocation',mod,rates,flush=True)
        ff,xx,_=data[mod,'train']; batch=data[mod,'validation']; f,z,vx=batch; m=source['model']
        idx=np.random.default_rng(m['training_draw_seed']).choice(len(ff),size=m['training_draws'],replace=True,p=hierarchy_weights(ff)); train=xx[idx]
        # Paired endpoints require re-playing the reference, not just matching panel counts.
        old=joblib.load(parent/f'{mod}_codec.joblib'); y=old.decode(old.encode(z,mode='full'),mode='full')
        re=reconstruction(f,z,y,vx,mod); par=summary['modalities'][mod]['operating_point']['bits_per_patch']
        hist=baseline[(baseline.modality==mod) & (baseline.method=='full') & (baseline.bits_per_patch==par) & (baseline.level=='subject')]
        a=re[re.level=='subject'].set_index(KEYS[:2]).x_nmse.sort_index(); b=hist.set_index(KEYS[:2]).x_nmse.sort_index()
        if not a.index.equals(b.index) or not np.allclose(a,b,atol=1e-9,rtol=1e-8): raise ValueError('parent per-subject X replay mismatch')
        prior_rates={4+i*j for i,j in m['rates'][mod]}; workers=c['resources']['workers']
        with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
            runs=joblib.Parallel(n_jobs=workers,pre_dispatch=workers)(joblib.delayed(run_one)(mod,r,parent,batch,train,prior_rates) for r in rates)
        ref=baseline[(baseline.modality==mod) & (baseline.method.isin(['flat','full','blind_full','legacy_hard'])) & (baseline.bits_per_patch.isin(rates))]
        table=pd.concat([ref]+[r[2] for r in runs],ignore_index=True); allrows.append(table)
        pts=table[(table.level=='macro') & (table.method=='adaptive_pattern')].sort_values('bits_per_patch')
        good=(pts.x_nmse<=.10) & (pts.centered_cosine>=.8)
        if mod=='eeg': good&=(pd.to_numeric(pts.band_30_45_db).abs()<=3) & (pd.to_numeric(pts.band_30_45_cosine)>=.70)
        row=pts[good].iloc[0] if good.any() else pts.iloc[-1]; bits=int(row.bits_per_patch)
        model=next(r[1] for r in runs if r[0]==bits); joblib.dump(model,out/f'{mod}_adaptive.joblib',compress=3)
        # Independently reload packed end-to-end interface on the first included site in each dataset.
        from_disk=joblib.load(out/f'{mod}_adaptive.joblib'); fixed=f[KEYS].drop_duplicates('dataset'); samples=[]
        for entry in fixed.to_dict('records'):
            pick=(f[KEYS]==pd.Series(entry)).all(axis=1).to_numpy(); sel=np.flatnonzero(pick)
            code=model.encode(z[sel],mode='conditional'); packed=model.pack(code,mode='conditional'); pred=from_disk.decode(from_disk.unpack(packed,len(sel),mode='conditional'),mode='conditional')
            np.testing.assert_array_equal(pred,model.decode(code,mode='conditional'))
            stem=entry['dataset']; filename=f'{mod}_{stem}_example.npz'; np.savez_compressed(out/filename,
                x=z[sel],pred=pred,patch=f.patch.to_numpy()[sel],pattern=code['pattern'],vx=vx[sel],payload=np.frombuffer(packed,np.uint8))
            samples.append(dict(**entry,file=filename,bits_per_patch=bits))
        contrasts=[paired_contrast(table,'x_nmse',ref,('adaptive_pattern',b)) for ref,b in [
            (('legacy_hard',14),14),(('flat',bits),bits),(('adaptive_blind',bits),bits),(('independent_tail',bits),bits),
            (('legacy_hard',14),bits)]]
        details[mod]=dict(selected_bits=bits,met_all_screens=bool(good.any()),x_nmse=float(row.x_nmse),
            centered_cosine=float(row.centered_cosine),examples=samples,contrasts=contrasts,
            per_subject_parent_replay_max_error=float((a-b).abs().max()),n_validation_patches=len(f),
            timings_s={r[0]:r[3] for r in runs})
        table.to_csv(out/f'{mod}_allocation.csv',index=False);dump(out/f'{mod}_decision.json',details[mod])
    pd.concat(allrows,ignore_index=True).to_csv(out/'allocation.csv',index=False)
    dump(out/'summary.json',dict(status='completed',experiment_id=c['experiment_id'],parent=str(parent.relative_to(ROOT)),
        modalities=details,scope='same previously examined development panel; no new task fitting or test',
        peak_controller_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=['synthetic','screen'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();c=configuration(args.config);out=args.output.resolve()
    if out.parent!=SUITE.resolve() or out==PARENT: raise ValueError('separate allocation output required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'allocation.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists(): raise ValueError('completed allocation run is immutable')
        if args.stage=='synthetic':
            if (out/'synthetic.json').exists(): raise ValueError('fresh run required')
            synthetic(); dump(out/'resolved_config.json',c); dump(out/'synthetic.json',gate(c));return
        if read(out/'synthetic.json')!=gate(c) or read(out/'resolved_config.json')!=c: raise ValueError('passed synthetic gate required')
        dump(out/'controller.json',dict(status='running',stage='paired_allocation'))
        try: screen(c,out)
        except Exception:
            (out/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc());dump(out/'controller.json',dict(status='failed'));raise
        dump(out/'controller.json',dict(status='completed',summary='summary.json'))


if __name__=='__main__': main()
