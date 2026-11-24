"""Replay paid tagged EEG words and render predeclared training voltage cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
from src.tokenizers.band_exchange_words import decode_native
from src.tokenizers.eeg_transient_gain import load_model as load_source
from audit_adaptive_band_exchange import render_review,write

STATE={}


def worker(dataset):
    run,hard,out,pop,rows,scales,selected=(STATE[k] for k in ['run','hard','out','population','rows','scales','selected'])
    cases=[c for c in pop['cases'] if c['dataset']==dataset]
    here=out/dataset;here.mkdir(exist_ok=True)
    all_examples=np.array([(c['crop']['row'],c['crop']['start_s'],p) for c in cases for p in c['crop']['positions']])
    index={tuple(e):j for j,e in enumerate(all_examples)}
    offsets={};pos=0
    for c in cases:offsets[c['crop']['row'],c['crop']['start_s']]=pos;pos+=c['count']
    lookup={(c['crop']['row'],c['crop']['start_s']):c for c in cases}
    results=[]
    for fold in range(4):
        subset=[c for c in selected if c['dataset']==dataset and lookup[c['row'],c['start_s']]['fold']==fold]
        if not subset:continue
        books={}
        for b in 'SO':
            books[b]=[]
            for role in range(5):
                with np.load(hard/'books'/f'fold_{fold}_{b}_{role}.npz') as z:
                    metadata=json.loads(str(z['metadata']))
                    if (metadata['fold'],metadata['branch'],metadata['role'])!=(fold,b,role):raise ValueError('wrong book fold')
                    books[b].append({k:z[k].copy() for k in z.files if k!='metadata'})
        with np.load(run/'component_source_transfer/adaptive_band_v1'/dataset/f'fold_{fold}_model.npz') as z:cov={b:z[b].copy() for b in 'SO'}
        source=load_source(run/'component_source_transfer/transient_gain_v1'/f'{dataset}_fold{fold}_transient.npz')
        if set(source.metadata['fit_subjects'])!={c['subject'] for c in cases if c['fold']!=fold}:raise ValueError('source outer people mismatch')
        with np.load(hard/dataset/f'fold_{fold}_ids.npz') as z:
            np.testing.assert_array_equal(z['examples'],all_examples)
            streams={arm:{b:z[arm+'_balanced_'+b].copy() for b in 'SO'} for arm in ['fixed','exchange']}
        with np.load(hard/dataset/f'fold_{fold}_metrics.npz') as z:
            metric_index={tuple(e):j for j,e in enumerate(z['examples'])}
            recorded={arm+'_'+method:z[arm+'_balanced_'+method+'__nmse'].copy() for arm in ['fixed','exchange'] for method in ['energy','allocated']}
        cache={}
        for case in subset:
            physical=lookup[case['row'],case['start_s']];key=case['row'],case['start_s'];which=case['row'],case['start_s'],case['position']
            if rows[case['row']]['split']!='train' or physical['subject']!=case['subject']:raise ValueError('train only before reads')
            if key not in cache:
                start=offsets[key];indices=np.arange(start,start+physical['count']);inverse=np.argsort(physical['native'])
                cache[key]={}
                for arm in ['fixed','exchange']:
                    paid={b:streams[arm][b][indices[inverse]] for b in 'SO'}
                    if any(v.dtype.kind not in 'ui' or v.max()>5115 for v in paid.values()):raise ValueError('paid integer range')
                    for method in ['energy','allocated']:
                        cache[key][arm+'_'+method]=decode_native(paid,dataset,books,cov,source,'balanced',method)
                cache[key]['paid']={arm:{b:streams[arm][b][indices[inverse]] for b in 'SO'} for arm in ['fixed','exchange']}
            ex=np.array([which]);batch=base_tokenizer_batch(rows,ex,scales)
            native=physical['native'][physical['crop']['positions'].index(case['position'])]
            arrays={k:batch[k][0] for k in ['x','s','o','mask']}
            checks={}
            method=case['method']
            for arm in ['fixed','exchange']:
                pred={k:v[native:native+1] for k,v in cache[key][arm+'_'+method].items()}
                scores=joint_reconstruction_metrics(pred,batch,1,np.array([case['scale']]))['metrics']['nmse'][0]
                expected=recorded[arm+'_'+method][metric_index[which]]
                np.testing.assert_array_equal(scores,expected)
                checks[arm]=float(np.abs(scores-expected).max())
                for b,v in pred.items():arrays[arm+'_'+b]=v[0]
                for b,v in cache[key]['paid'][arm].items():arrays[arm+'_ids_'+b]=v
            path=here/f"example_{case['audit_index']:03d}.npz";np.savez_compressed(path,**arrays)
            label='actual K5116 integer decode · balanced '+method
            results.append(dict(case,array=str(path.relative_to(out)),fold=fold,score_replay_gap=checks,interface_label=label))
    write(here/'replay.json',dict(status='passed',examples=results,CPU_seconds=time.process_time(),decoder_input='paid S/O integers, frozen books, covariance, source only'))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();run=ROOT/args.run;hard=run/'component_source_transfer/band_exchange_words_v1';out=ROOT/args.output;out.mkdir(exist_ok=True,parents=True)
    if args.render_only:render_review(run,out);return
    pop=json.loads((run/'component_source_transfer/hard_as_v1/local_information_v1/population.json').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()),ROOT);rows=manifest['rows']
    if (pop['cfg_identity'],pop['scales_identity'])!=(manifest['cfg_identity'],manifest['scales_identity']):raise ValueError('G0 coordinate mismatch')
    original=json.loads((run/'component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json').read_text())['examples']
    selected={}
    def add(case,rule,method='energy'):
        key=(case['row'],case['start_s'],case['position'],method)
        if key not in selected:
            clean={k:v for k,v in case.items() if k!='nmse'}
            selected[key]=dict(clean,method=method,audit_index=len(selected),audit_selection=[rule])
        else:selected[key]['audit_selection'].append(rule)
    for c in original:add(c,'original_'+','.join(c['selection']))
    for dataset in sorted(pop['native_positions']):
        ex=[];nmse=[]
        for fold in range(4):
            path=hard/dataset/f'fold_{fold}_metrics.npz'
            if not path.is_file():raise FileNotFoundError('wait for completed scoring before hard visual selection: '+str(path))
            with np.load(path) as z:ex.append(z['examples']);nmse.append(z['exchange_balanced_energy__nmse'])
        ex=np.concatenate(ex);nmse=np.concatenate(nmse)
        for label,value in [('new_hard_joint_worst',nmse.mean((1,2))),('new_hard_S_worst',nmse[:,1,0])]:
            i,t,p=map(int,ex[np.argmax(value)]);row=rows[i]
            case=dict(dataset=dataset,subject=row['subject'],record=row['record'],parent=row['parent'],row=i,start_s=t,position=p,position_name=row['positions'][p],scale=scales['positions']['eeg'][dataset][row['positions'][p]]['scale'],source_unit=row['source_unit'])
            add(case,label)
    # Matched allocated readout has a separate view for each old hard dataset case.
    for dataset in sorted(pop['native_positions']):
        hard_case=next(c for c in original if c['dataset']==dataset and 'candidate_max' in c['selection'])
        add(hard_case,'original_hard_allocated_readout_separate','allocated')
    write(out/'selected_examples.json',dict(examples=list(selected.values()),selection='All 17 original G1 ranked fixed/random/P90/old hard examples, new exchange-balanced-energy joint/S worst, plus four allocated-readout hard controls. No development data.'))
    STATE.update(run=run,hard=hard,out=out,population=pop,rows=rows,scales=scales,selected=list(selected.values()))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        records=[v for group in pool.map(worker,sorted(pop['native_positions'])) for v in group]
    records.sort(key=lambda c:c['audit_index'])
    write(out/'summary.json',dict(status='actual_paid_integer_display_replay_passed',stage='single_frequency_tagged_ID',is_single_ID_codec=True,interface_description='actual paid K5116 integer words, 13 fixed bits per ID, frequency role paid inside ID',codebook_K=5116,fixed_width_bits_per_ID=13,EEG_two_branch_bits_per_native_position_second=13,examples=records,confirmation_reads=0,test_reads=0,validation_reads=0))
    render_review(run,out)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
