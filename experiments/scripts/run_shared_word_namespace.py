"""Fit and evaluate a common physical word spelling across decodable native roles."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys,json
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluate_joint_tokenizer import write
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit,feature_means
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.tokenizers.shared_word_namespace import fit_alignment,compact,expand_native,STRATA,WIDTH
from src.utils.io import save_npz_atomic

STATE={};FIELDS={'theta':(0,1),'alpha':(1,2),'RMS':(2,10)}


def run_fold(fold):
    out,cfg,run,bank,pop=(STATE[k] for k in ['out','cfg','run','bank','pop'])
    fit=bank['fold']!=fold;held=~fit;valid=bank['valid'];fv=fit[:,None]&valid;hv=held[:,None]&valid
    if set(bank['subject'][fit])&set(bank['subject'][held]):raise ValueError('person leakage')
    lookup={tuple(e):j for j,e in enumerate(bank['examples'])};saved={};metrics={};mappings={};proof={}
    for model in cfg['models']:
        original={b:np.full(valid.shape,-1,np.int32) for b in 'SO'}
        for ds in sorted(pop['native_positions']):
            for stage,expected in [('fit',True),('held',False)]:
                with np.load(run/cfg['model']/ds/f'{stage}_fold{fold}_ids.npz') as z:
                    ix=np.array([lookup[tuple(e)] for e in z['examples']])
                    assert np.all(fit[ix]==expected) and np.all(original['S'][ix]==-1)
                    for b in 'SO':original[b][ix]=z[model+'_'+b]
        for b in 'SO':assert np.all(original[b]>=0)
        for scheme in cfg['schemes']:
            maps={};encoded={}
            for b in 'SO':
                spelling=np.tile(np.arange(1,WIDTH+1,dtype=np.uint16),(STRATA,1))
                if scheme=='shared_signature':
                    spelling=fit_alignment(original[b][fv],bank['background'][fv],bank[b+'_attributes'][fv],cfg['seed']+fold*2+(b=='O'))
                label=f'{model}_{scheme}_{b}';maps[b]=spelling;mappings[label]=spelling
                words=compact(original[b],spelling);encoded[b]=words;saved[label]=words
                head=SparsePatternAudit(WIDTH+1,4*len(pop['background_sites']),[],10,FIELDS)
                head.fit(words[fv],bank['background'][fv],bank[b+'_attributes'][fv])
                code=words[hv];bg,_,pred=head.read(code,bank['background'][hv]);y=bank[b+'_attributes'][hv]
                tested=valid[held];window=np.zeros((*tested.shape,2,3))
                window[tested]=np.stack([feature_means((w-y)**2,FIELDS) for w in (bg,pred)],axis=1)
                v=window.sum(1)/tested.sum(1).clip(1)[:,None,None];v[~tested.any(1)]=np.nan
                people=np.zeros(WIDTH+1,int)
                for p in np.unique(bank['subject'][fit]):
                    mask=valid&(bank['subject']==p)[:,None];people[np.unique(words[mask])]+=1
                ok=(head.local.count[code]>=8)&(people[code]>=2)
                metrics[label+'__errors']=v;metrics[label+'__qualified']=np.array([ok.sum(),len(ok)])
                metrics[label+'__fit_count']=head.local.count;metrics[label+'__fit_people']=people;metrics[label+'__held_count']=np.bincount(code,minlength=WIDTH+1)
            payloads=0
            for c in pop['cases']:
                take=np.array([lookup[c['crop']['row'],c['crop']['start_s'],p] for p in c['crop']['positions']])[np.argsort(c['native'])]
                # The packed transport includes only 11 bits per physical ID.
                streams={}
                for b in 'SO':
                    words=encoded[b][take];raw=np.packbits(((words.ravel()[:,None]>>np.arange(11))&1).astype(np.uint8).ravel(),bitorder='little')
                    bits=np.unpackbits(raw,bitorder='little',count=11*words.size).reshape(-1,11)
                    streams[b]=(bits@np.left_shift(1,np.arange(11))).astype(np.uint16).reshape(words.shape)
                restored=expand_native(streams,maps)
                for b in 'SO':np.testing.assert_array_equal(restored[b],original[b][take])
                payloads+=1
            proof[model+'_'+scheme]=dict(fit_and_held_arrays=payloads,every_original_word_bitwise_recovered=True)
    save_npz_atomic(out/f'fold_{fold}_maps.npz',**mappings)
    save_npz_atomic(out/f'fold_{fold}_ids.npz',examples=bank['examples'],**saved)
    save_npz_atomic(out/f'fold_{fold}_metrics.npz',examples=bank['examples'][held],**metrics)
    write(out/f'fold_{fold}_proof.json',proof)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());run=ROOT/cfg['run'];a.output.mkdir(parents=True,exist_ok=True)
    if cfg['fit']!='other_original_training_people_only' or any(cfg[k]!='forbidden' for k in ['validation','confirmation','test']):raise ValueError('restricted training-person protocol required')
    bankdir=run/cfg['bank']
    with np.load(bankdir/'training_attributes.npz') as z:bank=dict(z)
    pop=json.loads((bankdir/'population.json').read_text())
    g0=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest=base_tokenizer_training_data(g0,ROOT)[0];rows=manifest['rows']
    assert len(bank['examples'])==17040 and len(pop['cases'])==400 and all(rows[i]['split']=='train' for i,t,pos in bank['examples'])
    STATE.update(out=a.output,cfg=cfg,run=run,bank=bank,pop=pop)
    write(a.output/'resolved_config.json',cfg)
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(run_fold,range(4)))
    with np.load(a.output/'fold_0_metrics.npz') as z:labels=[k.split('__')[0] for k in z.files if k.endswith('__errors')]
    output={}
    for label in labels:
        ex=[];scores=[];support=np.zeros(2,int)
        for f in range(4):
            with np.load(a.output/f'fold_{f}_metrics.npz') as z:
                ex.append(z['examples']);scores.append(z[label+'__errors']);support+=z[label+'__qualified']
        h=base_training_aggregate({'nmse':np.concatenate(scores)},np.concatenate(ex),rows)['metrics']['nmse']
        mean=np.asarray(h['macro']);gain=1-mean[1]/mean[0]
        people=np.sum([np.asarray(v['value'])[1]<np.asarray(v['value'])[0] for v in h['by_subject']],axis=0)
        output[label]=dict(gain=gain,support=support[0]/support[1],positive_people=people,hierarchy=h,passed=bool(np.all(gain[:2]>0) and max(gain)>=.05 and min(people[:2])>=30 and support[0]/support[1]>=.9))
    write(a.output/'summary.json',dict(actual_single_word=output,rate_bits=11,decoded_original_IDs_exact=True,held_people=60,held_sites=17040,validation_reads=0,confirmation_reads=0,test_reads=0))


if __name__=='__main__':main()
