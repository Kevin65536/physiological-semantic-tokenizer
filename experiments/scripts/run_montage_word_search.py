"""Test joint selection of native waveform words on the existing training-person folds."""
import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch, base_tokenizer_joint_eeg_context
from src.tokenizers.band_exchange_words import load_fold_books, source_properties
from src.tokenizers.cross_branch_innovations import decode_words, paid_fields, pilot_positions, PILOT_SIZE
from src.tokenizers.eeg_transient_gain import load_model
from src.tokenizers.montage_word_search import dictionaries, local_spectrum, spectral_residual, select_words, synthesis_matrix
from src.utils.io import save_npz_atomic

STATE={}
BRANCHES=('S','O')



def training_folds(cfg):
    run=ROOT/cfg['run']
    pop=json.loads((run/cfg['population']).read_text())
    g0=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(g0,ROOT)
    if (pop['cfg_identity'],pop['scales_identity'])!=(manifest['cfg_identity'],manifest['scales_identity']):
        raise ValueError('frozen training coordinates changed')
    rows=manifest['rows']
    for c in pop['cases']:
        row=rows[c['crop']['row']]
        if (row['split'],row['dataset'],row['subject'])!=('train',c['dataset'],c['subject']):
            raise ValueError('only pre-existing training cases allowed')
    return run,pop,rows,scales


def case_arrays(sent, books, table, state, source, dataset, x, seen, refs):
    n=len(seen); predicted=source.predict_continuous(x[None],dataset,seen[None])
    target=np.concatenate([predicted[b][0] for b in BRANCHES]+[x])
    props={b:source_properties(predicted[b][0]) for b in BRANCHES}
    _,_,roles=paid_fields(sent,*books,state['scale'],'innovation',False)
    matrix=synthesis_matrix(roles,state)
    weight=np.concatenate([np.array([1/np.ravel(v['target_variance'][b])[0] for v in refs])*seen for b in ('s','o','x')])
    if not np.all(np.isfinite(weight)) or np.any(weight<0):raise ValueError('invalid frozen training weight')
    return target,props,roles,matrix,weight


def run_dataset(dataset, fold):
    cfg,run,pop,rows,scales,out,stage=(STATE[k] for k in ('cfg','run','pop','rows','scales','out','stage'))
    cases=[c for c in pop['cases'] if c['dataset']==dataset]
    orig=run/cfg['frozen'];dest=out/dataset;dest.mkdir(exist_ok=True)
    names=pop['native_positions'][dataset]; ref={k.upper():v for k,v in scales['positions']['eeg'][dataset].items()};refs=[ref[k.upper()] for k in names]
    cache=OrderedDict();before=time.process_time()
    score_file=dest/f'{stage}_fold{fold}_scores.npz';id_file=dest/f'{stage}_fold{fold}_ids.npz'
    if score_file.is_file():return
    with np.load(orig/dataset/f'fold_{fold}_ids.npz') as z:
        ex=z['examples'];primary={b:z['innovation_'+b] for b in BRANCHES};group=z['fold']
    old={b:load_fold_books(run/cfg['pilot'],fold,b) for b in BRANCHES}
    second={b:load_fold_books(orig/'books',fold,b) for b in BRANCHES}
    with np.load(orig/dataset/f'fold_{fold}_state.npz') as z:state=dict(z)
    if set(state['fit_people']) & set(state['held_people']):raise ValueError('fitted person overlaps held fold')
    books=(old,second);table=dictionaries(*books);source=load_model(run/cfg['source']/f'{dataset}_fold{fold}_transient.npz')
    if set(source.metadata['fit_subjects'])!=set(state['fit_people']):raise ValueError('source fold differs')
    arms=cfg['arms']
    event_mode='event_cap' in cfg
    if event_mode:
        from src.tokenizers.native_event_words import decode_events, pulse_shapes, replace_events
        pulses=pulse_shapes()
        event_scales=np.sqrt([np.ravel(v['target_variance'][b])[0] for b in ('s','o') for v in refs])
    warm={}
    if 'warm_search' in cfg and stage!='replay':
        part='fit' if stage=='fit' else 'held'
        with np.load(run/cfg['warm_search']/dataset/f'{part}_fold{fold}_ids.npz') as archive:
            warm=dict(archive)
    saved={}
    if stage=='replay':
        with np.load(dest/f'held_fold{fold}_ids.npz') as z:saved=dict(z)
    metrics=defaultdict(list);ids=defaultdict(list);cost=[];offset=0;held_offset=0
    for c in cases:
        take=np.arange(offset,offset+c['count']);offset+=len(take)
        if (group[take]!=c['fold']).any():raise ValueError('original fold identity changed')
        if (c['fold']==fold)==(stage=='fit'):continue
        native=np.array(c['native']);order=take[native.argsort()]
        np.testing.assert_array_equal(native[native.argsort()],np.arange(len(native)))
        example=ex[take];ids['examples'].append(example)
        sent={b:primary[b][order] for b in BRANCHES};codes={'frozen':sent}
        if stage=='replay':
            ix=(np.arange(held_offset,held_offset+len(take)))[native.argsort()]
            np.testing.assert_array_equal(saved['examples'][ix],ex[order])
            for arm in arms[1:]:codes[arm]={b:saved[arm+'_'+b][ix] for b in BRANCHES}
        else:
            x,seen,_=base_tokenizer_joint_eeg_context(rows,c['crop'],scales,names,0,allowed_splits=('train',))
            x=x.T.reshape(len(names),15,400).astype(np.float32)
            target,props,roles,matrix,weight=case_arrays(sent,books,table,state,source,dataset,x,seen,refs)
            for arm in arms[1:]:
                family=arm.split('_')[0]
                limit=64 if family=='semantic64' else 0
                initial=sent
                if warm:
                    ix=np.arange(held_offset,held_offset+len(take))[native.argsort()]
                    np.testing.assert_array_equal(warm['examples'][ix],ex[order])
                    initial={b:warm[family+'_'+b][ix] for b in BRANCHES}
                if arm.endswith('_prior'):
                    codes[arm]=initial
                    continue
                if arm.endswith('_event'):
                    codes[arm],record=replace_events(initial,table,state,event_scales,target,weight,pulses,cfg['event_cap'],cfg.get('cap_rounding','ceil'))
                    cost.append(dict(row=c['crop']['row'],start=c['crop']['start_s'],arm=arm,**record))
                    continue
                strength=cfg.get('magnitude_strength',0.) if arm.endswith('_spectrum') else 0.
                codes[arm],record=select_words(initial,matrix,target,weight,props,table,roles,limit,cfg['sweeps'],strength)
                cost.append(dict(row=c['crop']['row'],start=c['crop']['start_s'],arm=arm,**record))
                if stage=='probe':
                    pred=decode_words(codes[arm],*books,state,state,'innovation',False)
                    observed=np.concatenate([pred[k] for k in ('S','O','sum')])
                    _,loss=spectral_residual(target,observed,abs(local_spectrum(target)) if strength else None,strength)
                    np.testing.assert_allclose(record['loss'][-1],weight@loss,rtol=2e-11)
            for arm in arms[1:]:
                for b in BRANCHES:ids[arm+'_'+b].append(codes[arm][b][native])
        if stage!='fit':
            batch=base_tokenizer_batch(rows,example,scales,array_cache=cache,cache_limit=4)
            factors=input_scales(rows,example,scales)
            for arm,stream in codes.items():
                decoded=(decode_events(stream,table,state,event_scales,pulses) if arm.endswith('_event')
                         else decode_words(stream,*books,state,state,'innovation',False))
                preds={k:v[native] for k,v in decoded.items()}
                score=joint_reconstruction_metrics(preds,batch,1,factors)
                if score['support']['nonfinite_prediction_points'].any():raise ValueError('nonfinite wave')
                for k,v in score['metrics'].items():metrics[arm+'__'+k].append(v)
        held_offset+=len(take)
        if stage=='probe':break
    if stage!='replay':save_npz_atomic(id_file,**{k:np.concatenate(v) for k,v in ids.items()})
    full={k:np.concatenate(v) for k,v in metrics.items()}
    if stage=='replay':
        with np.load(dest/f'held_fold{fold}_scores.npz') as z:
            for k,v in full.items():np.testing.assert_array_equal(z[k],v)
    save_npz_atomic(score_file,examples=np.concatenate(ids['examples']),**full)
    write(dest/f'{stage}_fold{fold}_audit.json',dict(cpu_seconds=time.process_time()-before,stage=stage,fold=fold,cases=cost))
    print(dataset,fold,stage,'CPU',round(time.process_time()-before,2),flush=True)


def summarize():
    cfg,run,pop,rows,out=(STATE[k] for k in ('cfg','run','pop','rows','out'))
    fields=defaultdict(list)
    for d in sorted(pop['native_positions']):
        for f in range(4):
            with np.load(out/d/f'held_fold{f}_scores.npz') as z:
                for k in z.files:fields[k].append(z[k])
    ex=np.concatenate(fields.pop('examples'));assert len(ex)==17040
    arms={arm:base_training_aggregate({k.split('__',1)[1]:np.concatenate(v) for k,v in fields.items() if k.startswith(arm+'__')},ex,rows) for arm in cfg['arms']}
    base=json.loads((run/cfg['frozen']/'summary.json').read_text())['arms']['innovation_mean']['summary']['metrics']
    for k,v in base.items():np.testing.assert_allclose(arms['frozen']['metrics'][k]['macro'],v['macro'],rtol=1e-11,atol=1e-12,equal_nan=True)
    reference=json.loads((run/'component_source_transfer/integer_sum_source_v1/summary.json').read_text())['arms']['high_transient_source']['metrics']['nmse'];gates={}
    for arm,v in arms.items():
        m=v['metrics'];nm=np.ravel(m['nmse']['macro']);ref=np.ravel(reference['macro']);gain=np.ravel(m['rms_gain']['macro'])
        gates[arm]=dict(X=nm[0]<=1.05*ref[0],SO=all(nm[1:]<=1.1*ref[1:]),datasets=all(np.all(np.asarray(x)<=1.1*np.asarray(reference['by_dataset'][d])) for d,x in m['nmse']['by_dataset'].items()),RMS=np.all((gain>=.75)&(gain<=1.25)),bands=all(np.max(np.abs(m[k]['macro']))<=3.0103 for k in ('band_4_8_db','band_4_8_abs_db','band_8_13_db','band_8_13_abs_db')),correlation=min(np.ravel(m['correlation']['macro']))>0,envelope=min(np.ravel(m['local_rms_correlation']['macro']))>=.5)
    write(out/'summary.json',dict(arms=arms,waveform_gates=gates,arrays=400,position_windows=17040,people=60,zero_validation_confirmation_test_reads=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['probe','held','fit','replay','summary'],required=True);a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text())
    if cfg['data']!={'fit':'train','score':'train_held_people','validation':'forbidden','confirmation':'forbidden','test':'forbidden'}:raise ValueError('training-only contract required')
    run,pop,rows,scales=training_folds(cfg);out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    STATE.update(cfg=cfg,run=run,pop=pop,rows=rows,scales=scales,out=out,stage=a.stage)
    saved=out/'resolved_config.json'
    if saved.exists() and json.loads(saved.read_text())!=cfg:raise ValueError('retained run config differs')
    write(saved,cfg)
    if a.stage=='summary':summarize()
    else:
        datasets=sorted(pop['native_positions'],key=lambda d:-sum(c['count']**2 for c in pop['cases'] if c['dataset']==d))
        tasks=[(d,f) for d in datasets for f in (range(1) if a.stage=='probe' else range(4))]
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(run_dataset,d,f) for d,f in tasks]
            for job in jobs:job.result()


if __name__=='__main__':main()
