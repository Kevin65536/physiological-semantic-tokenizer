"""Independent train-only single actual-ID readout for complementary S/O sampling."""
from __future__ import annotations

import argparse,csv,hashlib,json,multiprocessing,os,subprocess,sys,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit,feature_means
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_training_data

FIELDS={'power_4_8':(0,1),'power_8_13':(1,2),'rms_envelope':(2,10)}
K=5116
STATE={}


def write(path,value):
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=scalar)+'\n')


def budget():
    unit=os.environ.get('CROSS_BRANCH_WORD_UNIT')
    if unit:
        value=subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True).strip()
        if value.isdigit() and int(value)>470e9:raise RuntimeError('480 CPU-second single-word audit budget reached; completed outputs retained')


def read_fold(bank,codes,fold,branch,backgrounds):
    valid=bank['valid'];fit=bank['fold']!=fold;held=~fit
    if codes.shape!=valid.shape or codes.dtype!=np.uint16 or np.any(codes>=K):
        raise ValueError('actual uint16 single-ID sequence within K5116 required')
    if set(bank['subject'][fit])&set(bank['subject'][held]):raise ValueError('person overlap across readout folds')
    train=valid&fit[:,None];test=valid[held]
    model=SparsePatternAudit(K,backgrounds,[],10,FIELDS)
    model.fit(codes[train],bank['background'][train],bank[branch+'_attributes'][train])
    ids,bg=codes[held][test],bank['background'][held][test]
    back,_,pred=model.read(ids,bg);truth=bank[branch+'_attributes'][held][test]
    error=np.zeros((*test.shape,2,3));error[test]=np.stack([feature_means((v-truth)**2,FIELDS) for v in [back,pred]],axis=1)
    result=error.sum(1)/test.sum(1).clip(1)[:,None,None];result[~test.any(1)]=np.nan
    usage={};person_counts={}
    for partition,rows in [('fit',fit),('held',held)]:
        mask=valid&rows[:,None];usage[partition]=np.bincount(codes[mask],minlength=K)
        counts=np.zeros(K,np.int32)
        for p in np.unique(bank['subject'][rows]):counts[np.unique(codes[valid&(bank['subject'][:,None]==p)])]+=1
        usage[partition+'_people']=counts
    support=(model.local.count[ids]>=8)&(usage['fit_people'][ids]>=2)
    mass=np.zeros(test.shape);mass[test]=support;mass=mass.sum(1)/test.sum(1).clip(1);mass[~test.any(1)]=np.nan
    return np.flatnonzero(held),result,mass,usage,dict(complete_held_patches=len(ids),qualified_held_patches=int(support.sum()),seen_ID_patches=int((model.local.count[ids]>0).sum()),seen_background_ID_patches=sum(model.joint.count.get(int(v),0)>0 for v in bg*K+ids),fit_people=np.unique(bank['subject'][fit]),held_people=np.unique(bank['subject'][held]))


def synthetic():
    ids=np.broadcast_to(np.array([1,2]*7+[1],np.uint16),(6,15)).copy()
    bank=dict(valid=np.ones((6,15),bool),fold=np.array([0,0,0,1,1,1]),subject=np.arange(6),background=np.zeros((6,15),int),S_attributes=np.repeat(ids[...,None].astype(float),10,axis=2),fixed_band=np.arange(6)%5)
    take,error,mass,usage,count=read_fold(bank,ids,0,'S',1)
    np.testing.assert_array_equal(take,[0,1,2]);np.testing.assert_array_equal(error[:,1],0);np.testing.assert_array_equal(mass,1)
    assert usage['fit'][1]==24 and usage['fit'][2]==21 and len(usage['fit'])==K
    assert usage['fit_people'][1]==3 and count['qualified_held_patches']==45
    try:read_fold(bank,ids.astype(float),0,'S',1)
    except ValueError:pass
    else:raise AssertionError('float IDs accepted')
    return dict(status='passed',same_actual_IDs_ignore_untransmitted_fixed_band=True,manual_fit_counts=[24,21],conditional_error_exact_zero=True,qualified_held_patches=45,K=K)


def audit_identity(run,reference,pop,bank,rows,out):
    arrays=defaultdict(dict);proof=[];dataset_rows={}
    if len(pop['cases'])!=400 or len({(c['dataset'],c['subject']) for c in pop['cases']})!=60:raise ValueError('original 400 arrays/60 people required')
    for c in pop['cases']:
        r=rows[c['crop']['row']]
        if r['split']!='train' or r['dataset']!=c['dataset'] or r['subject']!=c['subject']:raise ValueError('exact admitted training person before saved data')
    for f in range(4):
        budget()
        parts=defaultdict(list);ex=[];folds=[]
        for ds in sorted(pop['native_positions']):
            path=run/ds/f'fold_{f}_ids.npz'
            with np.load(path,allow_pickle=False) as z:new={k:z[k].copy() for k in z.files}
            with np.load(reference/ds/f'fold_{f}_ids.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in ['examples','fold','fixed_balanced_S','fixed_balanced_O']}
            np.testing.assert_array_equal(new['examples'],old['examples']);np.testing.assert_array_equal(new['fold'],old['fold'])
            for b in 'SO':np.testing.assert_array_equal(new['aligned_'+b],old['fixed_balanced_'+b])
            np.testing.assert_array_equal(new['aligned_S'],new['staggered_S'])
            ex.append(new['examples']);folds.append(new['fold'])
            cases=[c for c in pop['cases'] if c['dataset']==ds];offset=0;quota_checks=0
            for c in cases:
                count=c['count'];native=np.asarray(c['native']);order=np.argsort(native)
                np.testing.assert_array_equal(native[order],np.arange(len(pop['native_positions'][ds])))
                observed=(new['aligned_S'][offset:offset+count][order]>0).all(1);sites=np.flatnonzero(observed)
                original=np.where(observed,np.arange(len(observed))%5,-1);rotated=original.copy();rotated[sites]=np.roll(original[sites],2)
                for key in ['aligned_S','aligned_O','staggered_S','staggered_O']:
                    codes=new[key][offset:offset+count][order];tags=np.where(codes==0,-1,(codes.astype(np.int32)-1)//1023)
                    expected=rotated if key=='staggered_O' else original
                    np.testing.assert_array_equal(tags,np.broadcast_to(expected[:,None],tags.shape))
                    for t in range(15):np.testing.assert_array_equal(np.bincount(tags[observed,t],minlength=5),np.bincount(original[observed],minlength=5));quota_checks+=1
                offset+=count
            if offset!=len(new['examples']):raise ValueError('native case denominator changed')
            for key in ['aligned_S','aligned_O','staggered_S','staggered_O']:
                if new[key].dtype!=np.uint16 or new[key].shape!=(len(new['examples']),15) or np.any(new[key]>=K):raise ValueError('invalid paid-word interface')
                parts[key].append(new[key])
            proof.append(dict(dataset=ds,fold=f,positions=len(new['examples']),native_arrays=len(cases),aligned_fixed_actual_IDs_exact=True,S_layout_actual_IDs_exact=True,expected_paid_roles_exact=True,full_five_band_quota_checks=quota_checks,source_file=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        np.testing.assert_array_equal(np.concatenate(ex),bank['examples']);np.testing.assert_array_equal(np.concatenate(folds),bank['fold'])
        for key,values in parts.items():arrays[key][f]=np.concatenate(values)
    write(out/'identity_replay.json',dict(status='passed',proofs=proof,original_position_windows=len(bank['examples']),original_native_arrays=400,subjects=60,original_valid_two_second_patches=int(bank['valid'].sum()),no_hidden_role_namespace=True,source_scope='saved train-only actual full native IDs, all four source_f encodings fit and held'))
    return arrays


def task(key):
    arm,b=key.rsplit('_',1);bank,rows,pop,out,codes=(STATE[k] for k in ['bank','rows','pop','out','codes'])
    errors=np.empty((len(bank['fold']),2,3));mass=np.empty(len(bank['fold']));usage={};counts=[];dataset_usage={};control=[]
    ds_order=sorted(pop['native_positions']);dataset=np.array([rows[i]['dataset'] for i,t,p in bank['examples']])
    for f in range(4):
        budget();take,error,support,used,count=read_fold(bank,codes[key][f],f,b,4*len(pop['background_sites']))
        errors[take]=error;mass[take]=support;counts.append(dict(fold=f,**count))
        for name,value in used.items():usage[f'{name}_{f}']=value
        if arm=='aligned' or b=='S':
            with np.load(STATE['reference']/f'fixed_balanced_{b}_word_metrics.npz') as z:
                expected=z['word'][take]
                np.testing.assert_allclose(error,expected,rtol=8*np.finfo(float).eps,atol=0,equal_nan=True)
                for name in ['fit','fit_people','held','held_people']:np.testing.assert_array_equal(used[name],z[f'{name}_{f}'])
            control.append(dict(fold=f,original_control_count_exact=True,original_MSE_bitwise_equal=bool(np.array_equal(error,expected,equal_nan=True)),max_absolute_MSE_difference=float(np.nanmax(abs(error-expected))),max_relative_MSE_difference=float(np.nanmax(abs(error-expected)/np.maximum(abs(expected),1e-300)))))
        for part,selected in [('fit',bank['fold']!=f),('held',bank['fold']==f)]:
            patch=[];people=[]
            for ds in ds_order:
                mask=bank['valid']&selected[:,None]&(dataset==ds)[:,None];patch.append(np.bincount(codes[key][f][mask],minlength=K));persons=np.zeros(K,np.int32)
                for p in np.unique(bank['subject'][selected&(dataset==ds)]):persons[np.unique(codes[key][f][mask&(bank['subject'][:,None]==p)])]+=1
                people.append(persons)
            dataset_usage[f'{part}_by_dataset_{f}']=np.array(patch);dataset_usage[f'{part}_people_by_dataset_{f}']=np.array(people)
    hierarchy=base_training_aggregate({'nmse':errors},bank['examples'],rows)['metrics']['nmse'];m=np.asarray(hierarchy['macro']);gain=1-m[1]/m[0]
    persons=np.sum([np.array(v['value'])[1]<np.array(v['value'])[0] for v in hierarchy['by_subject']],axis=0)
    coverage=sum(c['qualified_held_patches'] for c in counts)/sum(c['complete_held_patches'] for c in counts)
    result=dict(gain=gain,people=persons,support=coverage,passed=bool(min(gain[:2])>0 and max(gain)>=.05 and min(persons[:2])>=30 and coverage>=.9),hierarchy=hierarchy,support_hierarchy=base_training_aggregate({'nmse':mass[:,None,None]},bank['examples'],rows)['metrics']['nmse'],support_counts=counts,control_replay=control,fold_usage=[])
    for f in range(4):
        fit,held,people=(usage[f'{k}_{f}'] for k in ['fit','held','fit_people']);order=np.argsort(-fit[1:],kind='stable')+1;cover=np.cumsum(fit[order])/max(1,fit[order].sum());rare=(fit<8)|(people<2)
        result['fold_usage'].append(dict(fold=f,fit_used_codes=int((fit[1:]>0).sum()),dead_nonmissing_codes=int((fit[1:]==0).sum()),held_used_codes=int((held[1:]>0).sum()),held_only_codes=int(((held[1:]>0)&(fit[1:]==0)).sum()),rare_fit_codes=int(rare[1:].sum()),held_patches_on_rare_codes=int(held[rare].sum()),fit_missing_ID0_patches=int(fit[0]),held_missing_ID0_patches=int(held[0]),minimum_major_codes_for_90percent=int(np.searchsorted(cover,.9)+1),major_code_order=order[:np.searchsorted(cover,.9)+1]))
    np.savez_compressed(out/f'{key}_word_metrics.npz',examples=bank['examples'],errors=errors,support=mass,dataset_order=np.array(ds_order),**usage,**dataset_usage)
    with (out/f'{key}_code_usage.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['fold','actual_ID','band_tag','reserved_missing','fit_patches','fit_people','held_patches','held_people'])
        for f in range(4):
            for k in range(K):w.writerow([f,k,(k-1)//1023 if k else -1,k==0,*[int(usage[f'{n}_{f}'][k]) for n in ['fit','fit_people','held','held_people']]])
    write(out/f'{key}_summary.json',result)
    return key,result


def render(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir=out/'figures_v2';figure_dir.mkdir(exist_ok=True)
    labels=['theta 4–8 Hz','alpha 8–13 Hz','8 quarter-second RMS'];ds_order=['eeg_fnirs_single_trial','refed','simultaneous_eeg_nirs','visual_cognitive_motivation']
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,b in enumerate('SO'):
        for a,arm in enumerate(['aligned','staggered']):
            key=arm+'_'+b;v=result[key];ax=axes[row,0];ax.bar(np.arange(3)+(a-.5)*.35,100*np.array(v['gain']),width=.35,color=['#3182bd','#e6550d'][a],label=arm)
            ax=axes[row,1]
            for ds,d in enumerate(ds_order):
                m=np.array(v['hierarchy']['by_dataset'][d]);g=1-m[1]/m[0]
                for j in range(3):ax.scatter(ds+(a-.5)*.22,100*g[j],marker=['o','s','^'][j],color=['#3182bd','#e6550d'][a],label=arm+' '+labels[j] if ds==0 else None)
        axes[row,0].set_xticks(range(3),labels);axes[row,0].set_title(f'EEG-{b}: same raw background + one actual ID');axes[row,0].set_ylabel('Gain over background (%)');axes[row,0].legend()
        axes[row,1].set_xticks(range(4),['SingleTrial','REFED','Simultaneous','Visual']);axes[row,1].set_title(f'EEG-{b}: all dataset-specific gains');axes[row,1].set_ylabel('Gain over same background (%)');axes[row,1].margins(y=.28);axes[row,1].legend(fontsize=7,ncol=2)
        for ax in axes[row]:ax.axhline(0,color='k',lw=.7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Cross-branch sampling actual-word audit: original 400 train arrays / 60 people\nFit other three person folds using their own source_f IDs; same teachers, masks and native backgrounds')
    fig.savefig(figure_dir/'actual_word_gains.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for col,b in enumerate('SO'):
        ax=axes[col]
        for j,arm in enumerate(['aligned','staggered']):
            v=result[arm+'_'+b];ax.bar(j,v['support'],color=['#3182bd','#e6550d'][j]);ax.text(j,v['support']+.01,f"{v['support']:.2%}",ha='center')
        ax.axhline(.9,color='#b30000',ls='--',label='original 90% support gate');ax.set_ylim(0,1.1);ax.set_xticks([0,1],['aligned','staggered']);ax.set_title('EEG-'+b);ax.set_ylabel('Qualified held patch fraction');ax.legend(fontsize=8)
    fig.suptitle('Support uses real paid words; each fold codebook is counted separately\nQualified: at least 8 complete fit patches and 2 fit people on the actual word')
    fig.savefig(figure_dir/'actual_word_support.png',dpi=220);plt.close(fig)
    table=[]
    for key,v in result.items():
        for level in ['by_dataset','by_subject']:
            entries=[dict(dataset=d,value=a) for d,a in v['hierarchy'][level].items()] if level=='by_dataset' else v['hierarchy'][level]
            for entry in entries:
                m=np.asarray(entry['value']);gain=1-m[1]/m[0];row=dict(arm=key,level=level,dataset=entry['dataset'],subject=entry.get('subject','all'))
                for name,val in zip(['theta','alpha','RMS'],gain):row[name+'_gain']=val
                table.append(row)
    with (out/'dataset_and_subject_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    write(out/'figure_manifest.json',dict(images=[str(figure_dir/'actual_word_gains.png'),str(figure_dir/'actual_word_support.png')],format='PNG',dpi=220,view_status='not_yet_opened',v2_changes='shorten support ylabel to prevent clipping, keep full definition in visible subtitle, add y margin above dataset dots for legend; all numeric values unchanged'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--output');p.add_argument('--synthetic',action='store_true');p.add_argument('--wait-seconds',type=int,default=0);p.add_argument('--render',action='store_true');args=p.parse_args()
    if args.synthetic:print(json.dumps(synthetic()));return
    if not args.config or not args.output:p.error('--config and --output required for saved train integer audit')
    cfg=yaml.safe_load((ROOT/args.config).read_text())
    if cfg['schema']!='cross_branch_sampling_v1' or cfg['data']!={'fit':'train','score':'train_held_people','development':'forbidden','confirmation':'forbidden','test':'forbidden'}:raise ValueError('fixed saved-train-only scope required before reads')
    base=ROOT/cfg['joint_run'];run=base/'component_source_transfer/cross_branch_sampling_v1';out=ROOT/args.output;out.mkdir(exist_ok=True,parents=True)
    if args.render:
        render(out,json.loads((out/'summary.json').read_text())['variants']);return
    pop=json.loads((base/cfg['population']).read_text());reference=base/cfg['dictionaries']
    deadline=time.monotonic()+args.wait_seconds
    while True:
        missing=[str(run/ds/f'fold_{f}_ids.npz') for ds in sorted(pop['native_positions']) for f in range(4) if not (run/ds/f'fold_{f}_ids.npz').is_file()]
        if not missing:break
        write(out/'progress.json',dict(status='waiting_for_parent_saved_integer_files',missing=missing))
        if time.monotonic()>=deadline:raise TimeoutError('parent saved integer tables not complete; metadata-only wait retained')
        time.sleep(2)
    write(out/'progress.json',dict(status='scoring_saved_train_actual_IDs',integer_files=16))
    manifest,_,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()),ROOT)
    with np.load(base/'component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz') as z:bank={k:z[k].copy() for k in z.files}
    protocol=dict(schema='cross_branch_actual_word_review_v1',experiment_id=cfg['experiment_id'],original_native_arrays=400,subjects=60,position_windows=17040,actual_ID_capacity=K,fit='other-three-person folds original raw means',background='original dataset/native-site/same-physical-local-X-RMS bin',features=FIELDS,held_scope='all original 400 arrays, matched same-fold source IDs for fit and held',primary_gate='theta>0 and alpha>0, max(theta,alpha,RMS)>=0.05, positive theta and alpha in>=30 of60 people, supported held patch mass>=0.90',support='>=8 complete train patches and>=2 train people per actual ID',control='aligned both branches actual IDs and metrics match fixed_balanced; S also equal staggered; all saved counts exact',hierarchy='position→crop→record→person→dataset equal',resources={'CPU_seconds':480,'workers':4,'threads':1,'memory_GiB':14},forbidden=['development','confirmation','test','new source encoding','waveform decoder changes'],parent_config=cfg)
    write(out/'resolved_word_protocol.json',protocol);(out/'source_execution.py').write_text(Path(__file__).read_text())
    codes=audit_identity(run,reference,pop,bank,manifest['rows'],out)
    STATE.update(bank=bank,rows=manifest['rows'],pop=pop,out=out,codes=codes,reference=reference)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:results=dict(pool.map(task,['aligned_S','aligned_O','staggered_S','staggered_O']))
    write(out/'summary.json',dict(status='completed_actual_single_word_audit',scope='same historical 400 train arrays/60 people only',variants=results,all_aligned_controls_replayed=True,staggered_S_same=True,single_word_numeric_pass={arm:all(results[arm+'_'+b]['passed'] for b in 'SO') for arm in ['aligned','staggered']},new_natural_context=('not_run_no_waveform_eligible_arm' if (run/'summary.json').exists() and not json.loads((run/'summary.json').read_text()).get('waveform_eligible') else 'deferred_until_parent_numeric_waveform_decision')))
    render(out,results)
    write(out/'progress.json',dict(status='completed_actual_single_word_audit',integer_files=16))
    print(json.dumps({key:{k:v[k] for k in ['gain','people','support','passed']} for key,v in results.items()},default=lambda v:v.tolist()))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
