"""One bounded train-crossfit HbO residual readout on unchanged saved integers."""
import argparse
from collections import defaultdict,OrderedDict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch
import yaml

from evaluate_joint_tokenizer import SOURCE,write,load_source,input_scales,display_selection
from run_dual_component_tokenizer import identity,owner_lock
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch,base_tokenizer_examples
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,joint_reconstruction_summary
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.tokenizers.contextual_latent_dictionary import load_model as load_base
from src.tokenizers.hb_residual_readout import features,apply_readout,load_model
from review_joint_tokenizer import score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranked(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def quiet_ids(model,ids):
    return (ids['S']==model.silent_ids['S']).all(1)&(ids['O']==model.silent_ids['O']).all(1)


def plan(rows,cfg):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    for i,t,p in base_tokenizer_examples(rows,'hb','train'):
        r=rows[i];groups[r['dataset']][r['subject']][r['record']][i,t].append(p)
    chosen=[];setting=cfg['training'];seed=setting['seed']
    for d,subjects in sorted(groups.items()):
        for s,records in sorted(subjects.items()):
            for r in sorted(records,key=lambda r:ranked([seed,d,s,r]))[:setting['records_per_subject']]:
                crops=records[r]
                for i,t in sorted(crops,key=lambda k:ranked([seed,d,s,r,*k]))[:setting['crops_per_record']]:
                    for p in sorted(crops[i,t],key=lambda p:ranked([seed,d,s,r,i,t,p]))[:setting['positions_per_crop']]:
                        chosen.append((i,t,p))
    return np.asarray(sorted(chosen),np.int32)


def collect(cfg,out,rows,scales,base):
    examples=plan(rows,cfg)
    if any(rows[i]['split']!='train' for i,_,_ in examples):raise ValueError('train boundary before arrays')
    cache=OrderedDict();parts=defaultdict(list);size=cfg['resources']['batch_size']
    for start in range(0,len(examples),size):
        take=examples[start:start+size];batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
        ids=base.encode(batch['x']);prediction=base.decode_ids(ids)
        for key,value in batch.items():parts[key].append(value)
        for key,value in prediction.items():
            if key in ('S','O','sum'):parts['prediction_'+key].append(value)
        for key,value in ids.items():parts['ids_'+key].append(value)
        parts['quiet'].append(quiet_ids(base,ids))
        write(out/'progress.json',dict(stage='train_only_collection',completed=start+len(take),planned=len(examples),CPU_seconds=time.process_time()))
    bank={key:np.concatenate(value) for key,value in parts.items()}
    np.savez_compressed(out/'training_bank.pending.npz',examples=examples,**bank)
    (out/'training_bank.pending.npz').replace(out/'training_bank.npz')
    write(out/'training_panel.json',dict(examples=examples,subjects=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples}),
        boundary='G0 train nonoverlapping admitted support; every training subject represented'))
    return examples,bank


def fit_linear(f,batch,examples,rows,grouping,ridge,indices):
    weights=training_weights(examples[indices],rows)
    names=np.asarray([rows[i]['dataset'] for i,_,_ in examples])
    groups=np.full(len(examples),'*',dtype=object) if grouping=='global' else names
    fitted={};normalizers={}
    mask=batch['mask'].reshape(len(examples),300,2)[:,:,0]
    y=batch['x'].reshape(len(examples),300,2)[:,:,0]-batch['prediction_sum'].reshape(len(examples),300,2)[:,:,0]
    variance=batch['vx'][:,0,0]
    for group in sorted(set(groups[indices])):
        local=indices[groups[indices]==group];w=weights[groups[indices]==group];w=w/w.sum()
        a=f[local];valid=mask[local];count=valid.sum(1)
        moment=np.where(valid[:,:,None],a*a,0.).sum(1)/count[:,None]
        scale=np.sqrt(np.einsum('n,nf->f',w,moment)).clip(1e-10)
        z=a/scale
        gram=np.einsum('n,ntf,ntg,nt->fg',w/variance[local]/count,z,z,valid,optimize=True)
        cross=np.einsum('n,ntf,nt,nt->f',w/variance[local]/count,z,y[local],valid,optimize=True)
        penalty=ridge*max(float(np.trace(gram)/len(scale)),1e-12)
        fitted[group]=(np.linalg.solve(gram+penalty*np.eye(len(scale)),cross)/scale).tolist()
        normalizers[group]=scale.tolist()
    return fitted,normalizers


def metric_summary(prediction,bank,examples,rows,scales):
    audited=joint_reconstruction_metrics(prediction,bank,2,input_scales(rows,examples,scales))
    return audited,joint_reconstruction_summary(audited['metrics'],audited['support'],examples.tolist(),rows)


def crossfit(cfg,out,rows,scales,examples,bank):
    prediction={key:bank['prediction_'+key] for key in ('S','O','sum')}
    original,reference=metric_summary(prediction,bank,examples,rows,scales)
    names=np.asarray([rows[i]['dataset'] for i,_,_ in examples]);folds=np.empty(len(examples),int)
    for dataset in sorted(set(names)):
        subjects=sorted({rows[i]['subject'] for i,_,_ in examples if rows[i]['dataset']==dataset},
                        key=lambda subject:ranked(['Hb-X-residual-CV-v1',dataset,subject]))
        for j,subject in enumerate(subjects):
            folds[[rows[i]['dataset']==dataset and rows[i]['subject']==subject for i,_,_ in examples]]=j%cfg['training']['folds']
    records=[];metric_arrays={};base_nmse=reference['metrics']['nmse'];setting=cfg['selection']
    for feature_set,grouping,ridge in itertools.product(cfg['model']['feature_sets'],cfg['model']['grouping'],cfg['model']['ridge_trace_fractions']):
        f=features(prediction,feature_set,cfg['model']['context_lag_samples'])
        f[bank['quiet']]=0.
        oof={key:np.asarray(value,float).copy() for key,value in prediction.items()};fit_records=[]
        for fold in range(cfg['training']['folds']):
            fit_indices=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
            coefficient,normalizers=fit_linear(f,bank,examples,rows,grouping,ridge,fit_indices)
            fitted=dict(feature_set=feature_set,grouping=grouping,context_lag_samples=cfg['model']['context_lag_samples'],coefficients=coefficient)
            held_prediction=apply_readout({key:value[held] for key,value in prediction.items()},bank['quiet'][held],names[held],fitted)
            for key,value in held_prediction.items():oof[key][held]=value
            fit_records.append(dict(fold=fold,held_subjects=sorted({(rows[examples[i,0]]['dataset'],rows[examples[i,0]]['subject']) for i in held}),coefficients=coefficient,feature_RMS=normalizers))
        audit,summary=metric_summary(oof,bank,examples,rows,scales)
        m=summary['metrics'];x=float(m['nmse']['macro'][0][0]);base_x=float(base_nmse['macro'][0][0]);improvement=1-x/base_x
        dataset_improvements={d:1-m['nmse']['by_dataset'][d][0][0]/base_nmse['by_dataset'][d][0][0] for d in base_nmse['by_dataset']}
        base_subject={(a['dataset'],a['subject']):a['value'][0][0] for a in base_nmse['by_subject']}
        improved=sum(a['value'][0][0]<base_subject[a['dataset'],a['subject']] for a in m['nmse']['by_subject'])
        visual='visual_cognitive_motivation'
        safety=[]
        for d in ['macro',*base_nmse['by_dataset']]:
            accessor=lambda key: np.asarray(m[key]['macro'] if d=='macro' else m[key]['by_dataset'][d])
            base_o=base_nmse['macro'][2][0] if d=='macro' else base_nmse['by_dataset'][d][2][0]
            safety.extend([accessor('nmse')[2,0]<=setting['maximum_CV_O_HbO_NMSE_ratio']*base_o,
                setting['rms_gain_interval'][0]<=accessor('rms_gain')[2,0]<=setting['rms_gain_interval'][1],
                accessor('boundary_jump_difference_nmse')[2,0]<=setting['boundary_difference_nmse_max'],
                accessor('slope_correlation')[2,0]>=setting['slope_correlation_min'],
                accessor('curvature_correlation')[2,0]>setting['curvature_correlation_min_exclusive']])
        eligible=bool(improvement>=setting['minimum_CV_X_HbO_relative_improvement'] and sum(v>0 for v in dataset_improvements.values())>=setting['minimum_improved_datasets']
            and dataset_improvements[visual]>0 and min(dataset_improvements.values())>=1-setting['maximum_dataset_CV_X_degradation_ratio']
            and improved/len(base_subject)>=setting['minimum_improved_subject_fraction'] and all(safety))
        key=f'{feature_set}__{grouping}__ridge{ridge:g}';free=f.shape[-1]*(1 if grouping=='global' else len(set(names)))
        records.append(dict(candidate=key,feature_set=feature_set,grouping=grouping,ridge=ridge,free_coefficients=free,
            CV_X_HbO_NMSE=x,relative_improvement=improvement,dataset_relative_improvements=dataset_improvements,
            improved_subjects=improved,subjects=len(base_subject),safety_passed=bool(all(safety)),eligible=eligible,
            folds=fit_records,metrics=summary))
        metric_arrays[key]=audit['metrics']['nmse']
        write(out/'progress.json',dict(stage='train_subject_crossfit',completed=len(records),planned=24,CPU_seconds=time.process_time()))
    eligible=[record for record in records if record['eligible']]
    selected=None
    if eligible:
        best=min(record['CV_X_HbO_NMSE'] for record in eligible)
        comparable=[record for record in eligible if record['CV_X_HbO_NMSE']<=setting['parsimony_X_objective_ratio']*best]
        selected=min(comparable,key=lambda record:(record['free_coefficients'],-record['ridge'],record['candidate']))
    np.savez_compressed(out/'crossfit_nmse.npz',examples=examples,folds=folds,source=original['metrics']['nmse'],**metric_arrays)
    write(out/'training_crossfit.json',dict(status='completed',source=reference,candidates=records,selected_candidate=None if selected is None else selected['candidate'],
        scope=cfg['training']['scope'],validation_arrays_read=0))
    if selected is None:return None
    f=features(prediction,selected['feature_set'],cfg['model']['context_lag_samples']);f[bank['quiet']]=0.
    coefficient,normalizers=fit_linear(f,bank,examples,rows,selected['grouping'],selected['ridge'],np.arange(len(examples)))
    fitted=dict(feature_set=selected['feature_set'],grouping=selected['grouping'],context_lag_samples=cfg['model']['context_lag_samples'],
        coefficients=coefficient,training_feature_RMS=normalizers,ridge=selected['ridge'])
    write(out/'selected.json',dict(status='frozen_before_any_screen',selection=selected['candidate'],fitted=fitted,
        allocation=cfg['model']['allocation'],free_coefficients=selected['free_coefficients']))
    return fitted


def read_summary(folder):
    for name in ('summary_units_v2.json','summary_v2.json','summary.json'):
        if (folder/name).exists():return json.loads((folder/name).read_text())
    raise FileNotFoundError(folder)


def export_figures(out,model,rows,scales,examples,ids,metrics,baseline):
    arguments=argparse.Namespace(reference_metrics=baseline/'metrics.npz')
    chosen=display_selection(arguments,metrics,rows,examples);indices=sorted(chosen)
    take=examples[indices];batch=base_tokenizer_batch(rows,take,scales)
    datasets=[rows[i]['dataset'] for i,_,_ in take]
    prediction=model.decode_ids({key:value[indices] for key,value in ids.items()},datasets)
    np.savez_compressed(out/'waveform_examples.npz',**{key:batch[key] for key in ('x','s','o','mask')},
        **{'prediction_'+key:value for key,value in prediction.items()})
    records=[]
    for n,index in enumerate(indices):
        i,t,p=examples[index];r=rows[i]
        records.append(dict(modality='hb',dataset=r['dataset'],subject=r['subject'],record=r['record'],parent=r['parent'],row=int(i),start_s=int(t),position=int(p),position_name=r['positions'][p],source_unit=r['source_unit'],scale=float(input_scales(rows,[examples[index]],scales)[0]),selection=chosen[index],panel_index=int(index),prediction_label='Integer decode',token_seconds=2,nmse=metrics['nmse'][index].tolist()))
    write(out/'waveform_examples.json',dict(coordinate='G0_normalized',model=str(model.metadata['checkpoint']),role=out.name,examples=records))


def evaluate(cfg,out,checkpoint,rows,scales,role):
    args=argparse.Namespace(panel=SOURCE/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1/panels.npz',modality='hb',role=role)
    checked_rows,checked_scales,examples,manifest=load_source(args)
    if identity(checked_scales)!=identity(scales) or identity(checked_rows)!=identity(rows):raise ValueError('panel G0 identity drift')
    source_ids=SOURCE/cfg['source']['integers']
    with np.load(source_ids) as stored:
        lookup={tuple(item):j for j,item in enumerate(stored['examples'])}
        selected=[lookup[tuple(item)] for item in examples]
        ids={key:stored[key][selected].copy() for key in ('S','O')}
    folder=out/role;folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'integer_ids.npz',examples=examples,**ids)
    with np.load(folder/'integer_ids.npz') as stored:ids={key:stored[key].copy() for key in ('S','O')}
    model=load_model(checkpoint);replay=load_model(checkpoint);model.metadata['checkpoint']=str(checkpoint)
    values=defaultdict(list);support=defaultdict(list);source_values=defaultdict(list);source_support=defaultdict(list);maximum=0.;quiet_count=0;cache=OrderedDict()
    for start in range(0,len(examples),cfg['resources']['batch_size']):
        take=examples[start:start+cfg['resources']['batch_size']];codes={key:value[start:start+len(take)] for key,value in ids.items()}
        datasets=[rows[i]['dataset'] for i,_,_ in take]
        batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
        base=model.source.decode_ids(codes);quiet=quiet_ids(model.source,codes)
        predicted=apply_readout(base,quiet,datasets,model.fitted);repeated=replay.decode_ids(codes,datasets)
        quiet_count+=int(quiet.sum())
        for key in ('S','O','sum'):
            np.testing.assert_array_equal(predicted[key],repeated[key]);maximum=max(maximum,float(np.max(abs(predicted[key]-repeated[key]))))
        np.testing.assert_array_equal(predicted['S'],base['S'])
        np.testing.assert_array_equal(predicted['O'].reshape(-1,300,2)[:,:,1],base['O'].reshape(-1,300,2)[:,:,1])
        for pred,collection,counts in ((repeated,values,support),(base,source_values,source_support)):
            audit=joint_reconstruction_metrics(pred,batch,2,input_scales(rows,take,scales))
            for key,value in audit['metrics'].items():collection[key].append(value)
            for key,value in audit['support'].items():counts[key].append(value)
        write(folder/'progress.json',dict(completed=start+len(take),planned=len(examples),CPU_seconds=time.process_time()))
    metrics={key:np.concatenate(parts) for key,parts in values.items()};counts={key:np.concatenate(parts) for key,parts in support.items()}
    summary=joint_reconstruction_summary(metrics,counts,examples.tolist(),rows)
    source_summary=joint_reconstruction_summary({key:np.concatenate(parts) for key,parts in source_values.items()},
        {key:np.concatenate(parts) for key,parts in source_support.items()},examples.tolist(),rows)
    summary.update(status='completed',checkpoint=str(checkpoint),modality='hb',role=role,
        integer_replay=dict(status='passed',independent_checkpoint_reload=True,metric_max_abs_difference=maximum),
        integer_identity='same saved source integers',quiet_windows=quiet_count,allocation=cfg['model']['allocation'])
    np.savez_compressed(folder/'metrics.npz',examples=examples,**metrics,**{'support_'+key:value for key,value in counts.items()})
    write(folder/'summary.json',summary);write(folder/'source_same_hardware.json',source_summary)
    run=out.parent.parent
    baseline=run/('baseline_audit/hb/vq128' if role=='screen' else 'development_audit/hb/vq128')
    continuous=run/('baseline_audit/hb/continuous' if role=='screen' else 'development_audit/hb/continuous')
    contract=yaml.safe_load((SOURCE/'experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml').read_text())
    decision=score(summary,read_summary(baseline),read_summary(continuous),contract)
    write(folder/'decision.json',decision)
    export_figures(folder,model,rows,scales,examples,ids,metrics,baseline)
    return decision


def run(args,cfg):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('completed bounded experiment retained')
    manifest,scales,_,_=base_tokenizer_training_data(cfg,SOURCE);rows=manifest['rows']
    source_path=SOURCE/cfg['source']['checkpoint'];source=load_base(source_path)
    if source.metadata['signature']['input']!=identity(manifest) or source.metadata['scales_identity']!=manifest['scales_identity']:raise ValueError('source/G0 mismatch')
    if 'decoder_component_transfer_gain' not in source.metadata or not hasattr(source,'silent_ids'):raise ValueError('frozen transfer/silence source required')
    write(out/'resolved_config.json',cfg)
    write(out/'duplicate_audit.json',dict(status='distinct',prior_transfer='preserves original X exactly',prior_G4_joint='different G4-1024 integers and neural decoders',this_experiment='train-crossfit linear X-HbO residual on fixed S128/O256 plus reserved pair'))
    examples,bank=collect(cfg,out,rows,scales,source)
    fitted=crossfit(cfg,out,rows,scales,examples,bank)
    if fitted is None:
        write(out/'summary.json',dict(status='training_crossfit_rejected',screen_read=False,development_read=False,CPU_seconds=time.process_time()));return
    shutil.copy2(source_path,out/'base.pt');shutil.copy2(source_path.parent/source.metadata['source_file'],out/source.metadata['source_file'])
    metadata=deepcopy(source.metadata);metadata.update(schema='hb_x_residual_v1',source_file='base.pt',source_sha256=digest(out/'base.pt'),
        residual_config_identity=identity(cfg),free_residual_coefficients=sum(len(value) for value in fitted['coefficients'].values()),allocation=cfg['model']['allocation'])
    checkpoint=out/'residual.pt';torch.save(dict(metadata=metadata,fitted=fitted),checkpoint)
    screen=evaluate(cfg,out,checkpoint,rows,scales,'screen')
    development=None
    if screen['reconstruction_passed']:development=evaluate(cfg,out,checkpoint,rows,scales,'development')
    write(out/'summary.json',dict(status='completed',checkpoint=str(checkpoint),screen=screen,
        development=development,confirm_read=False,test_read=False,CPU_seconds=time.process_time(),
        stage='single_train_selected_candidate_no_followup_branches'))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if cfg['schema']!='joint_hb_x_residual_v1':raise ValueError('versioned Hb-only residual experiment required')
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['resources']['process_CPU_limit_seconds'],)*2)
    torch.set_num_threads(cfg['resources']['torch_threads'])
    with owner_lock(args.output):
        try:run(args,cfg)
        except BaseException as error:
            write(args.output/'failure.json',dict(type=type(error).__name__,reason=str(error),CPU_seconds=time.process_time()));raise


if __name__=='__main__':main()
