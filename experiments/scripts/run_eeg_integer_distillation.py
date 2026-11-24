"""Bounded train-subject EEG low-ID sequence -> hard high-atom student diagnostic."""
from __future__ import annotations
import argparse
from collections import OrderedDict, Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tokenizers.eeg_integer_distillation import (integer_fields, fit_additive, neighbor_graph, load_model as load_student)
from src.tokenizers.spatial_waveform_dictionary import load_model as load_decoder, spectral_features
from src.tokenizers.eeg_component_transfer import IntegerSumSourceDecoder
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.analysis.dual_component_tokenizer import (joint_reconstruction_metrics, joint_reconstruction_summary, base_training_aggregate)
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch, base_tokenizer_joint_eeg

STATE = {}
FIELDS = {'power_4_8': (0, 1), 'power_8_13': (1, 2), 'rms_envelope': (2, 10)}
ARMS = ('low_original', 'current_ID_student', 'neighborhood_student', 'actual_high_teacher', 'continuous_source_upper')


def write(path, item):
    def default(a):
        if isinstance(a, np.ndarray): return a.tolist()
        if isinstance(a, np.generic): return a.item()
        raise TypeError(type(a).__name__)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=default)+'\n')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p, parent): return os.path.relpath(p, parent)
def reference(p, parent): return dict(path=relative(p,parent),sha256=sha(p))


def budget(*, review=False):
    cfg=STATE['cfg'];unit=os.environ.get('TOKENIZER_DISTILL_UNIT')
    used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    prior=float(os.environ.get('TOKENIZER_DISTILL_PRIOR_CPU','0'))
    if used+prior >= cfg['resources']['total_CPU_seconds']-(10 if review else 300):
        raise RuntimeError('fixed CPU budget reached; completed artifacts retained')
    return used+prior


def load_inputs(cfg):
    run=ROOT/cfg['run'];info=run/cfg['source']['local_information']
    hard=yaml.safe_load((ROOT/cfg['source']['contract']).read_text())
    pop=json.loads((info/'population.json').read_text());cases=pop['cases']
    with np.load(info/'training_attributes.npz',allow_pickle=False) as z: bank={k:z[k].copy() for k in z.files}
    manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text()),ROOT)
    rows=manifest['rows']
    if (len(cases),len(pop['subjects']),len(bank['examples']))!=(400,60,17040) or cfg['scope']['allowed_splits']!=['train']:
        raise ValueError('original train-only population required')
    for c in cases:
        r=rows[c['crop']['row']]
        if (r['dataset'],r['subject'],r['split'],r['modality'])!=(c['dataset'],c['subject'],'train','eeg'):
            raise ValueError('train boundary before any source ID reads')
        sl=slice(c['first'],c['first']+c['count'])
        np.testing.assert_array_equal(bank['examples'][sl],[(c['crop']['row'],c['crop']['start_s'],p) for p in c['crop']['positions']])
    if (pop['cfg_identity'],pop['scales_identity'])!=(manifest['cfg_identity'],manifest['scales_identity']):
        raise ValueError('G0 identity changed')
    return dict(cfg=cfg,run=run,info=info,hard=hard,pop=pop,cases=cases,bank=bank,rows=rows,scales=scales)


def load_codes(family, fold=None):
    info,run,cfg=STATE['info'],STATE['run'],STATE['cfg']
    result={};sources={}
    for dataset in STATE['pop']['native_positions']:
        if fold is None:
            paths=sorted((run/cfg['source']['oof_integers']/family/dataset).glob('ids_*.npz'))
            expected=[sha(run/STATE['hard']['families'][family][k]) for k in ('encoder','decoder')]
        else:
            paths=sorted((info/family/dataset/f'fold_{fold}').glob('fit_*.npz'))
            source=run/STATE['hard']['data']['source_model_run']/f'{dataset}_fold{fold}_transient.npz'
            sources[dataset]=sha(source)
            metadata=load_source(source).metadata
            fit_people={c['subject'] for c in STATE['cases'] if c['dataset']==dataset and c['fold']!=fold}
            if metadata['held_fold']!=fold or set(metadata['fit_subjects'])!=fit_people:
                raise ValueError('source ancestry includes held people')
        for p in paths:
            with np.load(p,allow_pickle=False) as z:
                if fold is None:
                    if z['identity'].tolist()!=expected: raise ValueError('root encoder/decoder mismatch')
                elif (int(z['source_fold'])!=fold or str(z['source_sha256'])!=sources[dataset]
                      or str(z['encoder_sha256'])!=sha(run/STATE['hard']['families'][family]['encoder'])):
                    raise ValueError('mixed fold source or encoder in fit stream')
                for j,key in enumerate(z['keys']):
                    key=(dataset,*map(int,key))
                    if key in result: raise ValueError('duplicate physical crop')
                    result[key]={b:z[('transient_'+b) if fold is None else b][j:j+1].copy() for b in ('S','O')}
                    if fold is not None:
                        result[key]['observed']=z['observed'][j].copy()
                        result[key]['X_sha256']=str(z['X_sha256'][j])
    expected={(c['dataset'],c['crop']['row'],c['crop']['start_s'],c['fold']) for c in STATE['cases'] if fold is None or c['fold']!=fold}
    if result.keys()!=expected: raise ValueError('original fit/OOF integer population incomplete')
    return result


def key(c): return c['dataset'],c['crop']['row'],c['crop']['start_s'],c['fold']


def crop_weights(cases, rows):
    people=defaultdict(set);records=defaultdict(set);crops=Counter()
    for c in cases:
        d,s=c['dataset'],c['subject'];r=rows[c['crop']['row']]['record']
        people[d].add(s);records[d,s].add(r);crops[d,s,r]+=1
    return {key(c):1/(len(people)*len(people[c['dataset']])*len(records[c['dataset'],c['subject']])*crops[c['dataset'],c['subject'],rows[c['crop']['row']]['record']]) for c in cases}


def fit_worker(task):
    fold,branch,band=task;out=STATE['out'];cfg=STATE['cfg'];budget()
    path=out/'books'/f'fold{fold}_{branch}_band{band}.npz'
    if path.exists(): return json.loads(path.with_suffix('.json').read_text())
    low,high=STATE['low'],STATE['high_source'];bank=STATE['bank']
    fit=[c for c in STATE['cases'] if c['fold']!=fold]
    weights=crop_weights(fit,STATE['rows'])
    l,h=load_codes('semantic',fold),load_codes('high',fold)
    low_feat=spectral_features(low.atoms[branch,band],band)
    high_feat=spectral_features(high.atoms[branch,band],band)
    xs,ys,ws=[],[],[]
    for c in fit:
        a,b=l[key(c)],h[key(c)]
        np.testing.assert_array_equal(a['observed'],b['observed'])
        if a['X_sha256']!=b['X_sha256']:raise ValueError('low/high fit X differ')
        fields=integer_fields({v:a[v] for v in ('S','O')},branch,STATE['graph'][c['dataset'],branch])[0]
        native=np.asarray(c['native']);sel=(native%5)==band
        if not sel.any():continue
        valid=bank['valid'][c['first']:c['first']+c['count']][sel]
        slots=native[sel]
        src=a[branch][0,slots];target=b[branch][0,slots]
        valid=valid&(src>0)&(target>0)
        xs.append(fields[slots][valid])
        ys.append(high_feat[target[valid]]-low_feat[src[valid]])
        weight=np.broadcast_to(weights[key(c)]/c['count']/np.maximum(bank['valid'][c['first']:c['first']+c['count']].sum(1)[sel,None],1),valid.shape)
        ws.append(weight[valid])
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    payload={};diagnostics={}
    for arm,current in [('current_ID_student',True),('neighborhood_student',False)]:
        state,diag=fit_additive(x,y,w,prior=cfg['model']['pseudo_count'],passes=cfg['model']['backfit_passes'],current_only=current)
        for name,val in state.items():payload[f'{arm}_{name}']=val
        diagnostics[arm]=diag
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**payload)
    result=dict(fold=fold,branch=branch,band=band,samples=len(x),fit_people=sorted({(c['dataset'],c['subject']) for c in fit}),
        held_people=sorted({(c['dataset'],c['subject']) for c in STATE['cases'] if c['fold']==fold}),
        teacher_fit_ancestry='same_source_f_both_families_never_OOF_from_other_folds',diagnostics=diagnostics,CPU_budget_used=budget())
    write(path.with_suffix('.json'),result);return result


def make_manifests():
    out=STATE['out'];run=STATE['run'];cfg=STATE['cfg']
    for fold in range(4):
        for arm in ('current_ID_student','neighborhood_student'):
            payload={}
            for b in ('S','O'):
                for band in range(5):
                    with np.load(out/'books'/f'fold{fold}_{b}_band{band}.npz',allow_pickle=False) as z:
                        for name in ('tables','intercept'):payload[f'{b}_{band}_{name}']=z[f'{arm}_{name}']
            for (d,b),value in STATE['graph'].items():payload[f'neighbors_{d}_{b}']=value
            state=out/'models'/f'fold{fold}_{arm}_state.npz';state.parent.mkdir(parents=True,exist_ok=True)
            if not state.exists():np.savez_compressed(state,**payload)
            for d in STATE['pop']['native_positions']:
                target=out/'models'/f'{d}_fold{fold}_{arm}.npz'
                meta=dict(schema=cfg['schema'],positions={d:STATE['pop']['native_positions'][d]},fold=fold,arm=arm,
                    transmitted_vocabulary=512,transmitted_bits_per_integer=9,internal_teacher_vocabulary=8192,
                    input_identity=STATE['pop']['cfg_identity'],scales_identity=STATE['pop']['scales_identity'],
                    low_decoder=reference(run/STATE['hard']['families']['semantic']['decoder'],target.parent),
                    high_decoder=reference(run/cfg['source']['integer_sum_decoder']/f'{d}_fold{fold}.npz',target.parent),
                    student_state=reference(state,target.parent),payload='low_S_O_only_no_teacher_IDs_no_X_no_local_scale')
                if not target.exists():np.savez_compressed(target,metadata=json.dumps(meta))


def metrics_one(waves,batch,scale):
    result=joint_reconstruction_metrics(waves,batch,1,scale)
    return result['metrics'],result['support']


def eval_worker(dataset,review=False):
    budget(review=review);out=STATE['out'];run=STATE['run'];cfg=STATE['cfg'];cases=[c for c in STATE['cases'] if c['dataset']==dataset]
    low_ids,high_ids=load_codes('semantic'),load_codes('high')
    cache=OrderedDict();rawcache=OrderedDict();metric_parts={a:defaultdict(list) for a in ARMS};support_parts={a:defaultdict(list) for a in ARMS};examples=[];identities=[]
    for fold in range(4):
        source=load_source(run/STATE['hard']['data']['source_model_run']/f'{dataset}_fold{fold}_transient.npz')
        low= IntegerSumSourceDecoder(STATE['low'],source,strength=1.)
        high=load_decoder(run/cfg['source']['integer_sum_decoder']/f'{dataset}_fold{fold}.npz')
        students={a:load_student(out/'models'/f'{dataset}_fold{fold}_{a}.npz') for a in ('current_ID_student','neighborhood_student')}
        for c in [c for c in cases if c['fold']==fold]:
            budget(review=review)
            payload=out/'payload'/dataset/f"{c['crop']['row']}_{c['crop']['start_s']}.npz";payload.parent.mkdir(parents=True,exist_ok=True)
            codes={b:low_ids[key(c)][b] for b in ('S','O')}
            if review:
                with np.load(payload,allow_pickle=False) as z:
                    for b in ('S','O'):np.testing.assert_array_equal(z[b],codes[b])
                    codes={b:z[b].copy() for b in ('S','O')}
            elif not payload.exists():np.savez_compressed(payload,**codes,key=np.asarray(key(c)[1:]),dataset=dataset,emitted_K=512)
            ex=STATE['bank']['examples'][c['first']:c['first']+c['count']]
            batch=base_tokenizer_batch(STATE['rows'],ex,STATE['scales'],array_cache=cache,cache_limit=16)
            scale=np.asarray([STATE['scales']['positions']['eeg'][dataset][STATE['rows'][i]['positions'][p]]['scale'] for i,_,p in ex])
            full,observed,_=base_tokenizer_joint_eeg(STATE['rows'],dict(c['crop'],positions=[]),STATE['scales'],STATE['pop']['native_positions'][dataset],rawcache,allowed_splits=('train',))
            value=full['x'].T.reshape(1,len(observed),15,400)
            waves={'low_original':low.decode_ids(codes,dataset),'actual_high_teacher':high.decode_ids({b:high_ids[key(c)][b] for b in ('S','O')},dataset),
                'continuous_source_upper':source.predict_continuous(value,dataset,observed[None])}
            for a,m in students.items():waves[a]=m.decode_ids(codes,dataset)
            hashes={a:{b:hashlib.sha256(w[b].tobytes()).hexdigest() for b in ('S','O','sum')} for a,w in waves.items()}
            for a,w in waves.items():
                pred={b:w[b][0,c['native']] for b in ('S','O','sum')}
                met,sup=metrics_one(pred,batch,scale)
                for k,v in met.items():metric_parts[a][k].append(v)
                for k,v in sup.items():support_parts[a][k].append(v)
            examples.append(ex);identities.append(dict(key=key(c),hashes=hashes,low_integers_sha256=sha(payload)))
            if len(identities)%8==0:write(out/f'progress_{dataset}.json',dict(stage='independent_replay' if review else 'hard_evaluation',crops=len(identities),planned=len(cases),CPU_budget_used=budget(review=review)))
    arrays={f'{a}__{k}':np.concatenate(v) for a,data in metric_parts.items() for k,v in data.items()}
    arrays.update({f'{a}__support__{k}':np.concatenate(v) for a,data in support_parts.items() for k,v in data.items()})
    arrays['examples']=np.concatenate(examples)
    path=out/f'metrics_{dataset}.npz'
    if review:
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(arrays):raise ValueError('metric schema changed')
            for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
        expected=json.loads((out/f'prediction_hashes_{dataset}.json').read_text())
        if expected!=json.loads(json.dumps(identities)):raise ValueError('independent decoded wave hashes differ')
    else:
        np.savez_compressed(path,**arrays);write(out/f'prediction_hashes_{dataset}.json',identities)
    result=dict(dataset=dataset,position_windows=len(arrays['examples']),crops=len(cases),all_payloads_reloaded=review,
        exact_all_arm_prediction_hashes_and_metrics=review,CPU_budget_used=budget(review=review))
    write(out/f'{"review" if review else "evaluation"}_{dataset}.json',result);return result


def raw_semantics():
    bank,rows=STATE['bank'],STATE['rows'];backgrounds=len(STATE['pop']['background_sites'])*4
    held=load_codes('semantic');codes={b:np.zeros(bank['valid'].shape,np.int32) for b in ('S','O')}
    for c in STATE['cases']:
        for b in ('S','O'):codes[b][c['first']:c['first']+c['count']]=held[key(c)][b][0,c['native']]
    result={};errors={};support_result={}
    for branch in ('S','O'):
        error=np.zeros((len(bank['valid']),2,3));mass=np.zeros(len(bank['valid']));counts=[]
        for fold in range(4):
            fit=bank['fold']!=fold;hv=bank['valid'][~fit];values={k:v.copy() for k,v in codes.items()};fit_codes=load_codes('semantic',fold)
            for c in STATE['cases']:
                if c['fold']!=fold:
                    values[branch][c['first']:c['first']+c['count']]=fit_codes[key(c)][branch][0,c['native']]
            words=values[branch]+bank['fixed_band'][:,None]*512
            audit=SparsePatternAudit(2560,backgrounds,[],10,FIELDS);valid=bank['valid']&fit[:,None]
            audit.fit(words[valid],bank['background'][valid],bank[branch+'_attributes'][valid])
            people=np.zeros(2560,np.int32)
            for person in np.unique(bank['subject'][fit]):
                people[np.unique(words[bank['valid']&(bank['subject'][:,None]==person)])]+=1
            ids,bg=words[~fit][hv],bank['background'][~fit][hv]
            background,_,conditional=audit.read(ids,bg);target=bank[branch+'_attributes'][~fit][hv]
            e=np.zeros((*hv.shape,2,3));e[hv]=np.stack([feature_means((p-target)**2,FIELDS) for p in (background,conditional)],axis=1)
            error[~fit]=e.sum(1)/hv.sum(1).clip(1)[:,None,None];error[~fit&~bank['valid'].any(1)]=np.nan
            ok=(audit.local.count[ids]>=8)&(people[ids]>=2);m=np.zeros(hv.shape,bool);m[hv]=ok
            mass[~fit]=m.sum(1)/hv.sum(1).clip(1);counts.append(dict(fold=fold,complete=len(ids),supported=int(ok.sum())))
        h=base_training_aggregate({'nmse':error},bank['examples'],rows)['metrics']['nmse'];m=np.asarray(h['macro']);gain=1-m[1]/m[0]
        fraction=np.mean([np.asarray(p['value'])[1]<np.asarray(p['value'])[0] for p in h['by_subject']],axis=0)
        coverage=sum(x['supported'] for x in counts)/sum(x['complete'] for x in counts)
        with np.load(STATE['info']/'semantic_transient_readout_errors.npz',allow_pickle=False) as old:
            np.testing.assert_array_equal(old[branch+'_error'],error);np.testing.assert_array_equal(old[branch+'_mass'],mass)
        result[branch]=dict(gain=dict(zip(FIELDS,gain)),positive_subject_fraction=dict(zip(FIELDS,fraction)),support_mass=coverage,
            hierarchy=h,support_counts=counts,passed=bool(min(gain[:2])>0 and max(gain)>=.05 and min(fraction[:2])>=.5 and coverage>=.9),
            unchanged_actual_low_ID_errors_bitwise_match_previous_pure_fold_audit=True)
        errors[branch+'_error']=error;errors[branch+'_mass']=mass
    np.savez_compressed(STATE['out']/'raw_low_word_information.npz',**errors,examples=bank['examples'])
    write(STATE['out']/'raw_low_word_information.json',result);return result


def summarize():
    all_parts=[]
    for d in STATE['pop']['native_positions']:
        with np.load(STATE['out']/f'metrics_{d}.npz',allow_pickle=False) as z:all_parts.append({k:z[k].copy() for k in z.files})
    ex=np.concatenate([p['examples'] for p in all_parts]);summary={}
    for a in ARMS:
        m={k.split('__',1)[1]:np.concatenate([p[k] for p in all_parts]) for k in all_parts[0] if k.startswith(a+'__') and '__support__' not in k}
        s={k.split('__support__')[1]:np.concatenate([p[k] for p in all_parts]) for k in all_parts[0] if k.startswith(a+'__support__')}
        summary[a]=joint_reconstruction_summary(m,s,ex,STATE['rows'])
    semantic=raw_semantics()
    lo=np.ravel(summary['low_original']['metrics']['nmse']['macro']);hi=np.ravel(summary['actual_high_teacher']['metrics']['nmse']['macro'])
    threshold=lo-.5*np.maximum(lo-hi,0)
    candidate=summary['neighborhood_student'];v=np.ravel(candidate['metrics']['nmse']['macro'])
    mechanism=bool(np.all(v<=threshold))
    dataset_pass={d:bool(np.all(np.ravel(candidate['metrics']['nmse']['by_dataset'][d])<=1.10*np.ravel(summary['low_original']['metrics']['nmse']['by_dataset'][d]))) for d in candidate['metrics']['nmse']['by_dataset']}
    decisions=dict(mechanism_threshold_X_S_O=threshold,neighborhood_macro_X_S_O=v,half_low_to_high_gap_all_components=mechanism,
        dataset_component_tolerance=dataset_pass,raw_transmitted_low_pattern_all_branches=all(r['passed'] for r in semantic.values()),
        original_reconstruction_and_visual_gates='must_pass_independently_reviewed_after_summary',development_opened=False,
        original_K128_continuous_gate_not_replaced=True)
    final=dict(schema=STATE['cfg']['schema'],arms=summary,raw_word_information=semantic,decision=decisions,
        population=STATE['pop'],CPU_budget_used=budget(),validation_reads=0,confirmation_reads=0,test_reads=0,
        rate=dict(vocabulary=512,bits_each=9,integers_per_branch_position_30s=15,EEG_two_branch_bits_per_second_per_position=9,
            internal_teacher_IDs_transmitted=0,independent_decode_inputs='saved_low_S_O_only_fixed_model_and_native_layout'))
    write(STATE['out']/'summary.json',final)
    return final


def render(*, summary_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=STATE['out'];run=STATE['run'];cfg=STATE['cfg'];rows=STATE['rows']
    summary=json.loads((out/'summary.json').read_text());figdir=out/'figures_v1';figdir.mkdir(exist_ok=True)
    aliases=dict(eeg_fnirs_single_trial='ST',refed='REFED',simultaneous_eeg_nirs='Sim',visual_cognitive_motivation='Visual')
    colors={'target':'#161616','low_original':'#c27b1c','current_ID_student':'#9864a0','neighborhood_student':'#c44444','actual_high_teacher':'#087e8b','continuous_source_upper':'#83998d'}
    labels={'low_original':'K512 original','current_ID_student':'Current-ID student','neighborhood_student':'K512 context student','actual_high_teacher':'Actual K8192 teacher','continuous_source_upper':'Continuous X upper'}
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    names=['low_original','current_ID_student','neighborhood_student','actual_high_teacher','continuous_source_upper']
    for j,a in enumerate(names):
        v=np.ravel(summary['arms'][a]['metrics']['nmse']['macro'])
        ax[0,0].bar(np.arange(3)+(j-2)*.15,v,.15,label=labels[a],color=colors[a])
    ax[0,0].set(xticks=range(3),xticklabels=['X','S','O'],ylabel='Target-specific train-variance NMSE',title='400 crops / 60 original train people / four folds')
    ax[0,0].legend(fontsize=7)
    for j,a in enumerate(names):
        v=np.ravel(summary['arms'][a]['metrics']['rms_gain']['macro'])
        ax[0,1].plot(range(3),v,'o-',label=labels[a],color=colors[a])
    ax[0,1].axhspan(.75,1.25,color='gray',alpha=.15);ax[0,1].set(xticks=range(3),xticklabels=['X','S','O'],ylabel='Prediction RMS / target RMS',title='Original absolute amplitude interval')
    for j,a in enumerate(names):
        v=np.ravel(summary['arms'][a]['metrics']['local_rms_correlation']['macro'])
        ax[0,2].plot(range(3),v,'o-',color=colors[a])
    ax[0,2].axhline(.5,ls=':',color='gray');ax[0,2].set(xticks=range(3),xticklabels=['X','S','O'],ylabel='Quarter-second RMS correlation',title='Original timing/strength requirement')
    for col,b in enumerate(('S','O')):
        item=summary['raw_word_information'][b];v=100*np.array(list(item['gain'].values()))
        ax[1,col].bar(range(3),v,color='#596f85')
        for j,x in enumerate(v):ax[1,col].text(j,x,f'{x:.2f}%',ha='center',va='bottom' if x>=0 else 'top')
        ax[1,col].axhline(0,color='gray');ax[1,col].axhline(5,ls=':',color='gray')
        ax[1,col].set(xticks=range(3),xticklabels=['Theta','Alpha','Local RMS'],ylabel='Raw single-low-ID error reduction (%)',title=f'{b}: unchanged transmitted K512 | support {100*item["support_mass"]:.2f}%')
    grid=[];ynames=[]
    for d in sorted(aliases):
        for a in ('low_original','neighborhood_student','actual_high_teacher'):
            grid.append(np.ravel(summary['arms'][a]['metrics']['nmse']['by_dataset'][d]));ynames.append(aliases[d]+' / '+labels[a])
    im=ax[1,2].imshow(grid,aspect='auto',cmap='magma_r')
    for i,v in enumerate(grid):
        for j,x in enumerate(v):ax[1,2].text(j,i,f'{x:.3f}',ha='center',va='center',fontsize=7,color='white' if x>max(map(max,grid))*.6 else 'black')
    ax[1,2].set(yticks=range(len(ynames)),yticklabels=ynames,xticks=range(3),xticklabels=['X','S','O'],title='All dataset failures retained; no trimming')
    fig.colorbar(im,ax=ax[1,2],label='NMSE')
    fig.suptitle('K512 actual integers -> train-fitted internal hard K8192 atoms\nNo high IDs, X, per-window scales or continuous features transmitted. Only student/source estimation excludes each held fold.',fontsize=12)
    fig.supxlabel('Equal positions -> crop -> record -> subject -> dataset. Word gain = 1 − MSE(background+ID)/MSE(background); same original unshrunk readout.\nContinuous upper sees X and lies outside the discrete interface. Internal hard teacher IDs carry no additional transmitted bits.',fontsize=9)
    p=figdir/('training_summary_v2.png' if summary_only else 'training_summary.png');fig.savefig(p,dpi=190,bbox_inches='tight');plt.close(fig)
    if summary_only:return
    selections=[];rng=np.random.default_rng(cfg['training']['seed'])
    for d in sorted(aliases):
        with np.load(out/f'metrics_{d}.npz',allow_pickle=False) as z:
            ex=z['examples'].copy();loss=z['neighborhood_student__nmse'].mean(axis=(1,2))
        order=np.lexsort((ex[:,2],ex[:,1],ex[:,0]));finite=np.flatnonzero(np.isfinite(loss))
        chosen=[('first_sorted',int(order[0])),('fixed_rng',int(rng.choice(order))),('candidate_P90',int(finite[np.argmin(abs(loss[finite]-np.quantile(loss[finite],.9)))])),('candidate_worst',int(finite[np.argmax(loss[finite])]))]
        for rule,n in chosen:selections.append(dict(dataset=d,rule=rule,example=ex[n].tolist(),mean_XSO_NMSE=float(loss[n]),population_positions=len(ex)))
    write(figdir/'selection.json',dict(rule=cfg['evaluation']['figures'],samples=selections))
    cache=OrderedDict();rawcache=OrderedDict();records=[]
    for j,item in enumerate(selections):
        budget(review=True);d=item['dataset'];i,start,pos=item['example']
        c=next(c for c in STATE['cases'] if c['dataset']==d and c['crop']['row']==i and c['crop']['start_s']==start)
        f=c['fold'];source=load_source(run/STATE['hard']['data']['source_model_run']/f'{d}_fold{f}_transient.npz')
        with np.load(out/'payload'/d/f'{i}_{start}.npz',allow_pickle=False) as z:codes={b:z[b].copy() for b in ('S','O')}
        student=load_student(out/'models'/f'{d}_fold{f}_neighborhood_student.npz')
        low=IntegerSumSourceDecoder(STATE['low'],source,strength=1.)
        high=load_decoder(run/cfg['source']['integer_sum_decoder']/f'{d}_fold{f}.npz')
        highids=load_codes('high')[key(c)]
        waves={'neighborhood_student':student.decode_ids(codes,d),'low_original':low.decode_ids(codes,d),'actual_high_teacher':high.decode_ids({b:highids[b] for b in ('S','O')},d)}
        full,observed,_=base_tokenizer_joint_eeg(rows,dict(c['crop'],positions=[]),STATE['scales'],STATE['pop']['native_positions'][d],rawcache,allowed_splits=('train',))
        waves['continuous_source_upper']=source.predict_continuous(full['x'].T.reshape(1,len(observed),15,400),d,observed[None])
        batch=base_tokenizer_batch(rows,np.asarray([[i,start,pos]]),STATE['scales'],array_cache=cache,cache_limit=16)
        name=rows[i]['positions'][pos];native={v.upper():k for k,v in enumerate(STATE['pop']['native_positions'][d])}[name.upper()]
        scale=STATE['scales']['positions']['eeg'][d][name]['scale'];unit=rows[i].get('source_unit','native voltage')
        target={b:batch[k][0].ravel()*scale for b,k in [('S','s'),('O','o'),('sum','x')]};mask=batch['mask'][0].ravel();t=np.arange(6000)/200
        prediction={a:{b:w[b][0,native].ravel()*scale for b in target} for a,w in waves.items()}
        fig,axes=plt.subplots(3,3,figsize=(17,10),layout='constrained')
        for r,b in enumerate(('S','O','sum')):
            values=[target[b]]+[v[b] for v in prediction.values()];lo,hi=min(v.min() for v in values),max(v.max() for v in values);pad=max((hi-lo)*.04,1e-9)
            for col,(left,right) in enumerate(((0,30),(14,16),(13.5,14.5))):
                ax=axes[r,col]
                for a in ('continuous_source_upper','actual_high_teacher','low_original','neighborhood_student'):
                    ax.plot(t,prediction[a][b],color=colors[a],lw=.65,alpha=.9,label=labels[a])
                ax.plot(t,target[b],color=colors['target'],lw=.65,label='Supervised target')
                for edge in range(0,31,2):ax.axvline(edge,color='#a8a8a8',ls=':',lw=.5)
                ax.fill_between(t,lo-pad,hi+pad,where=~mask,color='steelblue',alpha=.15,step='post')
                ax.set(xlim=(left,right),ylim=(lo-pad,hi+pad),xlabel='Time in 30-s context (s)',ylabel=f'{"X = S+O" if b=="sum" else b} ({unit})')
                ax.set_title(['Full 200-Hz voltage','Complete 2-s token','Across 14-s token boundary'][col])
        axes[0,0].legend(fontsize=7,loc='upper right')
        fig.suptitle(f'{aliases[d]} / {c["subject"]} / record {rows[i]["record"]} / {name} / start {start}s / fold {f}\n{item["rule"]}, candidate mean XSO NMSE={item["mean_XSO_NMSE"]:.4g}; same y-axis per component; blue shade = masked; dotted = token edges',fontsize=11)
        fig.supxlabel('Physical voltage, no display smoothing, no target/reconstruction rescaling. Original targets and all selected failures retained.\nStudent receives saved K512 integers only; high and continuous are diagnostic controls with their distinct stated inputs.',fontsize=9)
        path=figdir/f'{j:02d}_{aliases[d]}_{item["rule"]}_voltage.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
        # Local strength and band timing preserve phase-insensitive physiology next to voltage.
        fig,axes=plt.subplots(3,3,figsize=(16,9),layout='constrained')
        frequency=np.fft.rfftfreq(400,1/200);window=np.hanning(400)
        def energy(v,band):
            z=v.reshape(15,400);z=z-z.mean(1,keepdims=True);power=abs(np.fft.rfft(z*window,axis=1))**2*2/(400*np.sum(window**2));return power[:,(frequency>=band[0])&(frequency<band[1])].sum(1)
        for r,b in enumerate(('S','O','sum')):
            allv={'target':target[b],**{a:v[b] for a,v in prediction.items()}}
            for a,v in allv.items():
                label='Supervised target' if a=='target' else labels[a]
                for col,band in enumerate(((4,8),(8,13))):axes[r,col].plot(np.arange(15)*2+1,energy(v,band),'-o',ms=2,color=colors[a],lw=.85,label=label)
                axes[r,2].plot(np.arange(120)*.25+.125,np.sqrt(np.mean(v.reshape(120,50)**2,axis=1)),color=colors[a],lw=.8,label=label)
            for col in range(3):
                axes[r,col].set(xlim=(0,30),xlabel='Time (s)',ylabel=(f'{b} energy ({unit}²)' if col<2 else f'{b} RMS ({unit})'),title=['Theta, 4–8 Hz','Alpha, 8–13 Hz','0.25-s local RMS'][col])
                for patch in np.flatnonzero(~mask.reshape(15,400).all(1)):
                    axes[r,col].axvspan(2*patch,2*patch+2,color='steelblue',alpha=.15)
        axes[0,0].legend(fontsize=7);fig.suptitle(f'{aliases[d]} / {c["subject"]} / {name} / {start}s | {item["rule"]}\nHann 2-s band energy; raw voltage remains in adjacent retained figure. Blue = incomplete mask; scoring uses complete patches.',fontsize=11)
        dynpath=figdir/f'{j:02d}_{aliases[d]}_{item["rule"]}_dynamics.png';fig.savefig(dynpath,dpi=160,bbox_inches='tight');plt.close(fig)
        records.append(dict(**item,subject=c['subject'],record=rows[i]['record'],native_name=name,source_fold=f,voltage=str(path.relative_to(ROOT)),dynamics=str(dynpath.relative_to(ROOT))))
        write(figdir/'rendered.json',records)
    return records


def direct_budget(*,review=False):
    unit=os.environ.get('TOKENIZER_DISTILL_DIRECT_UNIT');prior=float(os.environ.get('TOKENIZER_DISTILL_DIRECT_PRIOR_CPU','0'))
    used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    limit=STATE['direct_cfg']['resources']['total_CPU_seconds']-10 if review else STATE['direct_cfg']['resources']['main_stop_CPU_seconds']
    if used+prior>=limit:raise RuntimeError('fixed direct-synthesis supplement CPU budget reached; partial evidence retained')
    return used+prior


def direct_fit_worker(task):
    from src.tokenizers.eeg_integer_distillation import fit_full_atom_projection
    fold,branch,band=task;direct_budget();out=STATE['direct_out'];path=out/'books'/f'fold{fold}_{branch}_band{band}.npz'
    if path.exists():return json.loads(path.with_suffix('.json').read_text())
    fit=[c for c in STATE['cases'] if c['fold']!=fold];weights=crop_weights(fit,STATE['rows']);highids=load_codes('high',fold)
    count=np.zeros(8192);number=0
    for c in fit:
        native=np.asarray(c['native']);chosen=native%5==band
        valid=STATE['bank']['valid'][c['first']:c['first']+c['count']][chosen]
        ids=highids[key(c)][branch][0,native[chosen]]
        weight=np.broadcast_to(weights[key(c)]/c['count']/np.maximum(valid.sum(1)[:,None],1),valid.shape)
        count+=np.bincount(ids[valid],weights=weight[valid],minlength=8192);number+=int(valid.sum())
    count[0]=0;source=STATE['high_source'];features=spectral_features(source.atoms[branch,band],band)
    state,diag=fit_full_atom_projection(features,source.atoms[branch,band],count,ridge_fraction=STATE['direct_cfg']['projection']['ridge_fraction'])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**state,fit_weighted_occupancy=count)
    result=dict(fold=fold,branch=branch,band=band,fit_complete_patches=number,diagnostics=diag,CPU_budget_used=direct_budget())
    write(path.with_suffix('.json'),result);return result


def direct_manifests():
    out=STATE['direct_out'];base=STATE['out']
    for fold in range(4):
        arrays={}
        for b in ('S','O'):
            for k in range(5):
                with np.load(out/'books'/f'fold{fold}_{b}_band{k}.npz',allow_pickle=False) as z:
                    for f in ('coefficient','intercept'):arrays[f'{b}_{k}_{f}']=z[f]
        state=out/'models'/f'fold{fold}_full_atom_projection.npz';state.parent.mkdir(parents=True,exist_ok=True)
        if not state.exists():np.savez_compressed(state,**arrays)
        for d in STATE['pop']['native_positions']:
            for arm,old in [('neighborhood_direct','neighborhood_student'),('current_ID_direct','current_ID_student')]:
                p=out/'models'/f'{d}_fold{fold}_{arm}.npz'
                meta=dict(schema='eeg_low_integer_direct_full_atom_v1',student=reference(base/'models'/f'{d}_fold{fold}_{old}.npz',p.parent),projection=reference(state,p.parent),payload='actual_low_K512_S_O_only',positions={d:STATE['pop']['native_positions'][d]},fold=fold)
                if not p.exists():np.savez_compressed(p,metadata=json.dumps(meta))


def direct_eval_worker(dataset,review=False):
    from src.tokenizers.eeg_integer_distillation import load_direct_model,synthesize_full_atoms
    out=STATE['direct_out'];base=STATE['out'];run=STATE['run'];cfg=STATE['cfg'];cases=[c for c in STATE['cases'] if c['dataset']==dataset]
    names=('neighborhood_direct',) if review else tuple(STATE['direct_cfg']['arms'])
    parts={a:defaultdict(list) for a in names};support={a:defaultdict(list) for a in names};examples=[];hashes=[];routing=[]
    rootlow=load_codes('semantic');roothigh={} if review else load_codes('high');cache=OrderedDict()
    for fold in range(4):
        models={a:load_direct_model(out/'models'/f'{dataset}_fold{fold}_{a}.npz') for a in ('neighborhood_direct','current_ID_direct')}
        for j,c in enumerate(c for c in cases if c['fold']==fold):
            direct_budget(review=review)
            with np.load(base/'payload'/dataset/f"{c['crop']['row']}_{c['crop']['start_s']}.npz",allow_pickle=False) as z:ids={b:z[b].copy() for b in ('S','O')}
            for b in ('S','O'):np.testing.assert_array_equal(ids[b],rootlow[key(c)][b])
            waves={'neighborhood_direct':models['neighborhood_direct'].decode_ids(ids,dataset)}
            if not review:
                m=models['neighborhood_direct'];actual={b:roothigh[key(c)][b] for b in ('S','O')}
                waves['current_ID_direct']=models['current_ID_direct'].decode_ids(ids,dataset)
                inferred=m.student.infer_ids(ids,dataset)
                waves['neighborhood_hard_projected']=m.projection_control(ids,inferred,dataset)
                waves['true_high_projected_floor']=m.projection_control(ids,actual,dataset)
                if j==0:
                    source=STATE['high_source'];full={b:np.stack([source.atoms[b,p%5][actual[b][0,p]] for p in range(actual[b].shape[1])])[None] for b in ('S','O')}
                    via=synthesize_full_atoms(m.student.high,full,actual,dataset);original=m.student.high.decode_ids(actual,dataset)
                    for b in ('S','O','sum'):np.testing.assert_array_equal(via[b],original[b])
                    routing.append(dict(dataset=dataset,fold=fold,key=key(c),original_full_atom_route_bitwise_exact=True))
            ex=STATE['bank']['examples'][c['first']:c['first']+c['count']];batch=base_tokenizer_batch(STATE['rows'],ex,STATE['scales'],array_cache=cache,cache_limit=16)
            scale=np.asarray([STATE['scales']['positions']['eeg'][dataset][STATE['rows'][i]['positions'][p]]['scale'] for i,_,p in ex])
            for a,w in waves.items():
                met,sup=metrics_one({b:w[b][0,c['native']] for b in ('S','O','sum')},batch,scale)
                for k,v in met.items():parts[a][k].append(v)
                for k,v in sup.items():support[a][k].append(v)
            primary=waves['neighborhood_direct'];hashes.append(dict(key=key(c),waveform_hashes={b:hashlib.sha256(primary[b].tobytes()).hexdigest() for b in ('S','O','sum')}));examples.append(ex)
            if len(hashes)%8==0:write(out/f'progress_{dataset}.json',dict(stage='independent_primary_replay' if review else 'paired_full_atom_evaluation',completed=len(hashes),planned=len(cases),CPU_budget_used=direct_budget(review=review)))
    arrays={f'{a}__{k}':np.concatenate(v) for a,data in parts.items() for k,v in data.items()};arrays.update({f'{a}__support__{k}':np.concatenate(v) for a,data in support.items() for k,v in data.items()});arrays['examples']=np.concatenate(examples)
    path=out/f'metrics_{dataset}.npz'
    if review:
        with np.load(path,allow_pickle=False) as z:
            for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
        if json.loads((out/f'primary_hashes_{dataset}.json').read_text())!=json.loads(json.dumps(hashes)):raise ValueError('direct waveform replay hashes differ')
    else:
        np.savez_compressed(path,**arrays);write(out/f'primary_hashes_{dataset}.json',hashes);write(out/f'full_atom_route_{dataset}.json',routing)
    result=dict(dataset=dataset,crops=len(cases),position_windows=len(arrays['examples']),exact_primary_metrics_and_waveform_replay=review,CPU_budget_used=direct_budget(review=review))
    write(out/f'{"review" if review else "evaluation"}_{dataset}.json',result);return result


def direct_summary():
    out=STATE['direct_out'];all_parts=[]
    for d in STATE['pop']['native_positions']:
        with np.load(out/f'metrics_{d}.npz',allow_pickle=False) as z:all_parts.append({k:z[k].copy() for k in z.files})
    ex=np.concatenate([p['examples'] for p in all_parts]);arms={}
    for a in STATE['direct_cfg']['arms']:
        metrics={k.split('__',1)[1]:np.concatenate([p[k] for p in all_parts]) for k in all_parts[0] if k.startswith(a+'__') and '__support__' not in k}
        support={k.split('__support__')[1]:np.concatenate([p[k] for p in all_parts]) for k in all_parts[0] if k.startswith(a+'__support__')}
        arms[a]=joint_reconstruction_summary(metrics,support,ex,STATE['rows'])
    old=json.loads((STATE['out']/'summary.json').read_text());primary=arms['neighborhood_direct'];low=old['arms']['low_original'];high=old['arms']['actual_high_teacher']
    lo,hi=np.ravel(low['metrics']['nmse']['macro']),np.ravel(high['metrics']['nmse']['macro']);threshold=lo-.5*np.maximum(lo-hi,0);new=np.ravel(primary['metrics']['nmse']['macro'])
    gates={}
    for a,value in arms.items():
        m=value['metrics'];gain=np.ravel(m['rms_gain']['macro']);envelope=np.ravel(m['local_rms_correlation']['macro']);energy=[np.ravel(m[f'band_{b}_{e}_{s}']['macro']) for b,e in [(4,8),(8,13)] for s in ['db','abs_db']]
        gates[a]=dict(NMSE_half_gap_all=bool(np.all(np.ravel(m['nmse']['macro'])<=threshold)),gain_all=bool(np.all((gain>=.75)&(gain<=1.25))),theta_alpha_all=bool(max(float(np.max(abs(v))) for v in energy)<=3.0103),RMS_envelope_all=bool(np.all(envelope>=.5)),
            per_dataset_component_ratio_max=max(float(np.max(np.array(v)/np.array(low['metrics']['nmse']['by_dataset'][d]))) for d,v in m['nmse']['by_dataset'].items()))
    result=dict(arms=arms,reference_summary='../summary.json',decision=dict(primary='neighborhood_direct',same_base_half_gap_threshold=threshold,primary_NMSE=new,primary_gates=gates['neighborhood_direct'],all_arm_gates=gates,
        unchanged_raw_low_word_gates=old['decision']['raw_transmitted_low_pattern_all_branches'],development_opened=False,new_capacity_search=False),
        actual_low_payload_unchanged=True,projected_true_high_floor='full400_atom_regression_fit_only_never_assigned_band_only_waveforms',CPU_budget_used=direct_budget(),validation_reads=0,confirmation_reads=0,test_reads=0)
    write(out/'summary.json',result);return result


def direct_render():
    from src.tokenizers.eeg_integer_distillation import load_direct_model
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=STATE['direct_out'];base=STATE['out'];result=json.loads((out/'summary.json').read_text());old=json.loads((base/'summary.json').read_text())
    directory=out/'figures_v1';directory.mkdir(exist_ok=True)
    labels={'neighborhood_direct':'K512 direct neighbor','current_ID_direct':'K512 direct current','neighborhood_hard_projected':'K512 hard + projection','true_high_projected_floor':'True K8192 projection floor','low_original':'Original K512'}
    colors=dict(neighborhood_direct='#bd3f3f',current_ID_direct='#985da4',neighborhood_hard_projected='#d88d28',true_high_projected_floor='#11818b',low_original='#7c8586')
    arms=dict(result['arms'],low_original=old['arms']['low_original'])
    fig,ax=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for j,a in enumerate(labels):
        ax[0].bar(np.arange(3)+(j-2)*.15,np.ravel(arms[a]['metrics']['nmse']['macro']),.15,color=colors[a],label=labels[a])
        ax[1].plot(range(3),np.ravel(arms[a]['metrics']['rms_gain']['macro']),'o-',color=colors[a])
        ax[2].plot(range(3),np.ravel(arms[a]['metrics']['local_rms_correlation']['macro']),'o-',color=colors[a])
    for item in ax:item.set(xticks=range(3),xticklabels=['X','S','O'])
    ax[0].set(ylabel='G0 target-variance NMSE',title='Same actual K512 payload; four subject folds');ax[0].legend(fontsize=7)
    ax[1].axhspan(.75,1.25,color='gray',alpha=.15);ax[1].set(ylabel='Prediction / target RMS',title='Original absolute gain gate')
    ax[2].axhline(.5,ls=':',color='gray');ax[2].set(ylabel='0.25-s RMS correlation',title='Envelope timing is retained')
    fig.suptitle('Direct integer-conditioned full atoms remove the internal hard selection\n400 native train crops / 60 people / 17,040 position windows; all libraries and failures retained',fontsize=11)
    fig.supxlabel('Role coefficients -> fit-only full 400-point atom dictionary -> unchanged high spatial, local gain and source synthesis.\nTrue high projection floor is a privileged control with actual K8192 IDs. Primary direct decoder only receives K512 IDs.',fontsize=8)
    fig.savefig(directory/'summary.png',dpi=190,bbox_inches='tight');plt.close(fig)
    selections=[]
    for d in sorted(STATE['pop']['native_positions']):
        with np.load(out/f'metrics_{d}.npz',allow_pickle=False) as z:ex=z['examples'].copy();loss=z['neighborhood_direct__nmse'].mean(axis=(1,2))
        order=np.lexsort((ex[:,2],ex[:,1],ex[:,0]));chosen=[('direct_worst',int(np.nanargmax(loss)))]
        for rule,n in chosen:selections.append(dict(dataset=d,rule=rule,example=ex[n].tolist(),mean_XSO_NMSE=float(loss[n]),population=len(ex)))
    write(directory/'selection.json',dict(samples=selections,rule='primary_worst_each_dataset_fixed_four_negative_tail_cases_no_best_window_choice_resource_bounded'))
    rootlow=load_codes('semantic');roothigh={} if review else load_codes('high');cache=OrderedDict();records=[]
    for j,item in enumerate(selections):
        direct_budget(review=True);d=item['dataset'];row,start,p=item['example'];c=next(c for c in STATE['cases'] if c['dataset']==d and c['crop']['row']==row and c['crop']['start_s']==start);f=c['fold']
        ids={b:rootlow[key(c)][b] for b in ('S','O')};actual={b:roothigh[key(c)][b] for b in ('S','O')}
        model=load_direct_model(out/'models'/f'{d}_fold{f}_neighborhood_direct.npz');inferred=model.student.infer_ids(ids,d)
        source=model.student.high.estimator;low=IntegerSumSourceDecoder(STATE['low'],source,strength=1.)
        waves=dict(neighborhood_direct=model.decode_ids(ids,d),neighborhood_hard_projected=model.projection_control(ids,inferred,d),true_high_projected_floor=model.projection_control(ids,actual,d),low_original=low.decode_ids(ids,d))
        batch=base_tokenizer_batch(STATE['rows'],np.asarray([[row,start,p]]),STATE['scales'],array_cache=cache,cache_limit=16);name=STATE['rows'][row]['positions'][p]
        native={n.upper():k for k,n in enumerate(STATE['pop']['native_positions'][d])}[name.upper()];scale=STATE['scales']['positions']['eeg'][d][name]['scale'];unit=STATE['rows'][row].get('source_unit','native')
        target={b:batch[k][0].ravel()*scale for b,k in [('S','s'),('O','o'),('sum','x')]};pred={a:{b:v[b][0,native].ravel()*scale for b in target} for a,v in waves.items()};mask=batch['mask'][0].ravel();t=np.arange(6000)/200
        fig,axes=plt.subplots(3,4,figsize=(19,9),layout='constrained')
        for r,b in enumerate(('S','O','sum')):
            values=[target[b]]+[v[b] for v in pred.values()];lo=min(v.min() for v in values);hi=max(v.max() for v in values);pad=max(.04*(hi-lo),1e-10)
            for col,(left,right) in enumerate(((0,30),(14,16),(13.5,14.5))):
                ax=axes[r,col]
                for a in ('true_high_projected_floor','low_original','neighborhood_hard_projected','neighborhood_direct'):ax.plot(t,pred[a][b],color=colors[a],lw=.65,label=labels[a])
                ax.plot(t,target[b],color='black',lw=.65,label='Supervised target')
                ax.fill_between(t,lo-pad,hi+pad,where=~mask,color='steelblue',alpha=.12)
                for edge in range(0,31,2):ax.axvline(edge,color='gray',ls=':',lw=.4)
                ax.set(xlim=(left,right),ylim=(lo-pad,hi+pad),xlabel='Time in context (s)',ylabel=('X = S+O' if b=='sum' else b)+' (native)',title=['Full 200-Hz voltage','14–16 s local voltage','Across 14-s boundary'][col])
            ax=axes[r,3]
            for a in pred:ax.plot(np.arange(120)*.25+.125,np.sqrt(np.mean(pred[a][b].reshape(120,50)**2,axis=1)),color=colors[a],lw=.75)
            ax.plot(np.arange(120)*.25+.125,np.sqrt(np.mean(target[b].reshape(120,50)**2,axis=1)),color='black',lw=.75)
            ax.set(xlim=(0,30),xlabel='Time (s)',ylabel='RMS (native)',title='0.25-s local RMS')
        axes[0,0].legend(fontsize=6.5)
        fig.suptitle(f'{d} / {c["subject"]} / {STATE["rows"][row]["record"]} / {name} / {start}s / fold {f}\n{item["rule"]}: direct mean XSO NMSE={item["mean_XSO_NMSE"]:.4g}; native source unit={unit}; same target/prediction axes; mask blue; token edges dotted.',fontsize=10)
        fig.supxlabel('All plotted voltage samples retained, no smoothing or independent curve scaling. Direct real atoms are deterministic frozen functions of saved low integers.\nProjected true-high floor has high integers as an explicit control. No high integer or continuous atom is an input to the primary decoder.',fontsize=8)
        path=directory/f'{j:02d}_{d}_{item["rule"]}.png';fig.savefig(path,dpi=145,bbox_inches='tight');plt.close(fig)
        records.append(dict(**item,subject=c['subject'],position=name,fold=f,figure=str(path.relative_to(ROOT))));write(directory/'rendered.json',records)
    return records


def direct_main(args):
    cfg=yaml.safe_load((ROOT/args.direct_config).read_text());out=STATE['out']/cfg['output_suffix'];out.mkdir(parents=True,exist_ok=True);STATE.update(direct_cfg=cfg,direct_out=out)
    if cfg['base_contract']!=args.config or cfg['payload']!='unchanged_actual_K512_S_O_only':raise ValueError('same frozen low stream and base contract required')
    lock=(out/'producer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.render:direct_render();return
    if args.review:
        ctx=multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=cfg['resources']['workers'],mp_context=ctx) as pool:results=list(pool.map(direct_eval_worker,STATE['pop']['native_positions'],[True]*4))
        write(out/'independent_review.json',dict(passed=True,all400_primary_low_integer_only_full_waveform_hashes_and_all_metrics_exact=True,datasets=results,CPU_budget_used=direct_budget(review=True)));return
    if not (out/'resolved_config.json').exists():write(out/'resolved_config.json',dict(config=cfg,base_contract_sha256=sha(ROOT/args.config),source_sha256={p:sha(ROOT/p) for p in ['src/tokenizers/eeg_integer_distillation.py','experiments/scripts/run_eeg_integer_distillation.py']},preflight='6 targeted tests pass including full-atom synthesis identity and held-target-isolated fit'))
    ctx=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers'],mp_context=ctx) as pool:
        tasks=[(f,b,k) for f in range(4) for b in ('S','O') for k in range(5)];results=list(pool.map(direct_fit_worker,tasks))
    write(out/'fit_summary.json',results);direct_manifests()
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers'],mp_context=ctx) as pool:list(pool.map(direct_eval_worker,STATE['pop']['native_positions']))
    direct_summary()


def guarded_api_probe(args):
    """A fresh inference process forbids source integer/observation files while replaying four full native contexts."""
    import builtins
    import io
    from src.tokenizers.eeg_integer_distillation import load_direct_model
    cfg=yaml.safe_load((ROOT/args.config).read_text());run=ROOT/cfg['run'];base=run/cfg['output'];extra=yaml.safe_load((ROOT/args.direct_config).read_text());out=base/extra['output_suffix']
    selected=json.loads((out/'figures_v1/rendered.json').read_text());reads=set();checks=[]
    original_np,original_builtin,original_io=np.load,builtins.open,io.open
    def check(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(path).resolve()
            if (p.is_relative_to(ROOT/'data') or p.is_relative_to(run/cfg['source']['oof_integers']) or p.name.startswith(('ids_','fit_')) or p.name=='training_attributes.npz'):
                raise AssertionError('inference attempted teacher integers or observation/target arrays: '+str(p))
            if p.is_relative_to(ROOT):reads.add(str(p.relative_to(ROOT)))
    def guarded_np(path,*a,**kw):check(path);return original_np(path,*a,**kw)
    def guarded_open(path,*a,**kw):check(path);return original_builtin(path,*a,**kw)
    def guarded_io(path,*a,**kw):check(path);return original_io(path,*a,**kw)
    np.load,builtins.open,io.open=guarded_np,guarded_open,guarded_io
    try:
        for c in selected:
            d,f=c['dataset'],c['fold'];row,start,_=c['example']
            with np.load(base/'payload'/d/f'{row}_{start}.npz',allow_pickle=False) as z:ids={b:z[b].copy() for b in ('S','O')}
            model=load_direct_model(out/'models'/f'{d}_fold{f}_neighborhood_direct.npz');wave=model.decode_ids(ids,d)
            expected=json.loads((out/f'primary_hashes_{d}.json').read_text());ref=next(x for x in expected if x['key']==[d,row,start,f])
            got={b:hashlib.sha256(wave[b].tobytes()).hexdigest() for b in ('S','O','sum')}
            if got!=ref['waveform_hashes']:raise AssertionError('guarded pure inference hash mismatch')
            checks.append(dict(dataset=d,fold=f,example=c['example'],all_native_X_S_O_waveform_hashes_exact=True))
    finally:np.load,builtins.open,io.open=original_np,original_builtin,original_io
    write(out/'guarded_decoder_API_review.json',dict(passed=True,fresh_process=True,teacher_ID_and_raw_array_reads_forbidden=True,only_fixed_model_and_saved_low_payload_reads=sorted(reads),checks=checks,
        full400_numerical_replay='../direct_synthesis_v1/independent_review.json',scope='guarded_api_4_adverse_native_contexts_plus_all400_exact_numerical_replay'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--review',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--direct-config');p.add_argument('--render-summary',action='store_true');p.add_argument('--guarded-api',action='store_true');args=p.parse_args()
    if args.guarded_api:guarded_api_probe(args);return
    cfg=yaml.safe_load((ROOT/args.config).read_text());STATE.update(load_inputs(cfg));out=STATE['run']/cfg['output'];out.mkdir(parents=True,exist_ok=True);STATE['out']=out
    lock=(out/'producer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    threadpool_limits(1)
    STATE['low']=load_decoder(STATE['run']/STATE['hard']['families']['semantic']['decoder'])
    high=load_decoder(STATE['run']/STATE['hard']['families']['high']['decoder']);STATE['high_source']=getattr(high,'source',high)
    STATE['graph']=neighbor_graph(STATE['high_source'])
    if not args.review and not args.render and not args.direct_config and not args.render_summary:
        config_path=out/'resolved_config.json'
        if config_path.exists():
            if json.loads(config_path.read_text())['config']!=cfg:raise ValueError('retained config differs')
        else:
            write(config_path,dict(config=cfg,HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_files={v:sha(ROOT/v) for v in ('src/tokenizers/eeg_integer_distillation.py','experiments/scripts/run_eeg_integer_distillation.py',args.config)},
                affinity=list(os.sched_getaffinity(0)),supervisor=os.environ.get('TOKENIZER_DISTILL_UNIT'),preflight='tests_passed_no_AS_encodes',
                inherited_exposure=cfg['training']['inherited_exposure']))
    if args.direct_config:direct_main(args);return
    if args.render_summary:render(summary_only=True);return
    if args.prepare_only:return
    if args.render:render();return
    ctx=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers'],mp_context=ctx) as pool:
        if not args.review:
            futures=[pool.submit(fit_worker,(f,b,k)) for f in range(4) for b in ('S','O') for k in range(5)]
            results=[]
            for fut in as_completed(futures):
                results.append(fut.result());write(out/'fit_progress.json',dict(completed=len(results),planned=40,CPU_budget_used=budget()))
            write(out/'fit_summary.json',results)
    if not args.review:make_manifests()
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers'],mp_context=ctx) as pool:
        futures=[pool.submit(eval_worker,d,args.review) for d in STATE['pop']['native_positions']]
        results=[f.result() for f in as_completed(futures)]
    if args.review:write(out/'independent_review.json',dict(passed=True,exact_all400_payload_only_waveform_replay_and_metrics=True,datasets=results,CPU_budget_used=budget(review=True)))
    else:summarize()

if __name__=='__main__':main()
