#!/usr/bin/env python3
"""Fit two frozen O-color gains from training targets with the original S mapping retained."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import SOURCE,identity,write_json
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch,base_tokenizer_examples
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.tokenizers.contextual_latent_dictionary import load_model


def normal_equation(prediction,batch,input_scale):
    n = len(batch['x'])
    mask = batch['mask'].reshape(n,300,2)
    count = mask.sum(1)
    scale = np.asarray(input_scale)[:,None,None]
    p = prediction['O'].reshape(n,300,2).astype(float)*scale
    s = prediction['S'].reshape(n,300,2).astype(float)*scale
    y = batch['o'].reshape(n,300,2).astype(float)*scale
    x = batch['x'].reshape(n,300,2).astype(float)*scale
    vo = batch['vo'][:,0,:2]*scale[:,0,:]**2
    vx = batch['vx'][:,0,:2]*scale[:,0,:]**2
    def mean(value):
        return np.where(mask,value,0.).sum(1)/np.maximum(count,1)
    a = mean(p*p/vo[:,None,:]+p*p/vx[:,None,:])
    b = mean(p*y/vo[:,None,:]+p*(x-s)/vx[:,None,:])
    return np.stack((a,b),axis=1)


def transfer_moments(prediction,batch):
    """S'=source_sum-gO and O'=gO, using the exact frozen target masks/scales."""
    n=len(batch['x']);mask=batch['mask'].reshape(n,300,2)
    p=np.asarray(prediction['O'],float).reshape(n,300,2)
    s=np.asarray(prediction['sum'],float).reshape(n,300,2)-p
    ts,to=[batch[key].reshape(n,300,2).astype(float) for key in ('s','o')]
    vs,vo=[batch['v'+key][:,0,:2] for key in ('s','o')]
    def mean(value):return np.where(mask,value,0.).sum(1)/np.maximum(mask.sum(1),1)
    return dict(S_c=mean((s-ts)**2)/vs,S_b=mean(p*(s-ts))/vs,S_a=mean(p*p)/vs,
        O_c=mean((p-to)**2)/vo,O_b=mean(p*(p-to))/vo,O_a=mean(p*p)/vo,
        S_energy=mean(s*s),SP_energy=mean(s*p),P_energy=mean(p*p),
        S_target_energy=mean(ts*ts),O_target_energy=mean(to*to))


def development_family(args,cfg):
    """Diagnostic only: all positive gains satisfying fixed development NMSE/RMS constraints."""
    out=args.output
    if (out/'summary.json').exists():raise FileExistsError('completed development family retained')
    torch.set_num_threads(cfg['resources']['torch_threads'])
    manifest,scales,_,_=base_tokenizer_training_data(cfg,SOURCE);rows=manifest['rows']
    audit=args.development_source;baseline=args.development_baseline
    source_path=SOURCE/cfg['source_checkpoint'];model=load_model(source_path)
    if model.metadata['signature']['input']!=identity(manifest):raise ValueError('source G0 identity mismatch')
    panel_path=SOURCE/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1/panels.npz'
    panel_record=json.loads((panel_path.parent/'panel.json').read_text())
    with np.load(panel_path) as stored:examples=stored['hb_development'].copy()
    with np.load(audit/'integer_ids.npz') as stored:
        np.testing.assert_array_equal(stored['examples'],examples)
        codes={key:stored[key].copy() for key in ('S','O')}
    with np.load(baseline/'metrics.npz') as stored:np.testing.assert_array_equal(stored['examples'],examples)
    for i,_,_ in examples:
        row=rows[i]
        if row['split']!='validation' or row['subject']==panel_record['confirmation_subjects'][row['dataset']]:
            raise ValueError('full development excludes confirmation before arrays')
    original=json.loads((audit/'summary.json').read_text())
    if Path(original['checkpoint']).resolve()!=source_path.resolve():raise ValueError('integer source checkpoint mismatch')
    quiet=json.loads((SOURCE/cfg['quiet_training_fit']).read_text())
    collections={variant:{} for variant in ('model',*quiet['variants'])};cache=OrderedDict();changed={key:0 for key in collections}
    for start in range(0,len(examples),cfg['resources']['batch_size']):
        take=examples[start:start+cfg['resources']['batch_size']]
        batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=128)
        prediction=model.decode_ids({key:value[start:start+len(take)] for key,value in codes.items()})
        base=transfer_moments(prediction,batch)
        energy=np.square(batch['x'].astype(float)).mean((1,2))
        error=np.square(np.asarray(prediction['sum'],float)-batch['x']).mean((1,2))
        for variant in collections:
            moments={key:value.copy() for key,value in base.items()}
            fixed=np.zeros((len(take),2))
            if variant!='model':
                fit=quiet['variants'][variant];selected=(np.sqrt(energy)<=fit['threshold_input_sd'])&(energy<error)
                level=np.tile(fit['level_S'],20)[None,None,:]*np.ones_like(batch['x'])
                constant=transfer_moments(dict(S=level,O=-level,sum=np.zeros_like(level)),batch)
                for key in moments:moments[key][selected]=constant[key][selected]
                for key in ('S_b','S_a','O_b','O_a','SP_energy','P_energy'):moments[key][selected]=0.
                fixed[selected]=np.sqrt(constant['P_energy'][selected]/constant['O_target_energy'][selected])
                changed[variant]+=int(selected.sum())
            moments['O_fixed_gain']=fixed
            for key,value in moments.items():collections[variant].setdefault(key,[]).append(value)
        if start%2048==0:write_json(out/'progress.json',dict(stage='development_frozen_integer_moments',completed=start+len(take),planned=len(examples)))
    weights=training_weights(examples,rows);datasets=np.asarray([rows[i]['dataset'] for i,_,_ in examples])
    path=baseline/('summary_v2.json' if (baseline/'summary_v2.json').exists() else 'summary.json')
    reference=json.loads(path.read_text())['metrics']['nmse']
    results={}
    for variant,parts in collections.items():
        moments={key:np.concatenate(value) for key,value in parts.items()}
        np.savez_compressed(out/(variant+'_moments.npz'),examples=examples,**moments)
        def mean(value,selected=None):
            if selected is None:return np.einsum('n,nc->c',weights,value)
            return np.einsum('n,nc->c',weights[selected]/weights[selected].sum(),value[selected])
        totals={key:mean(value) for key,value in moments.items()}
        if variant=='model':
            np.testing.assert_allclose(np.stack([totals['S_c'],totals['O_c']]),
                np.asarray(original['metrics']['nmse']['macro'])[1:],rtol=1e-6,atol=1e-8)
        per_dataset={name:{key:mean(value,datasets==name) for key,value in moments.items()} for name in sorted(set(datasets))}
        colors=[]
        for color in range(2):
            ordinary=np.sqrt(moments['P_energy'][:,color]/moments['O_target_energy'][:,color])
            fixed=float(totals['O_fixed_gain'][color]);rate=float(weights@ordinary)
            interval=[max(0.,(.75-fixed)/rate),(1.25-fixed)/rate]
            constraints=[]
            for name,coef,bounds in [('macro',totals,1.05*np.asarray(reference['macro'])[1:,color]),
                *[(name,coef,1.1*np.asarray(reference['by_dataset'][name])[1:,color]) for name,coef in per_dataset.items()]]:
                for component,sign in [('S',-1),('O',1)]:
                    a=float(coef[component+'_a'][color]);b=2*sign*float(coef[component+'_b'][color])-2*a
                    c=float(coef[component+'_c'][color])-2*sign*float(coef[component+'_b'][color])+a
                    bound=float(bounds[0 if component=='S' else 1]);disc=b*b-4*a*(c-bound)
                    if a>0 and disc>=0:
                        roots=sorted(((-b-np.sqrt(disc))/(2*a),(-b+np.sqrt(disc))/(2*a)))
                    elif a<=0 and c<=bound:roots=[-float('inf'),float('inf')]
                    else:roots=[float('inf'),-float('inf')]
                    interval=[max(interval[0],roots[0]),min(interval[1],roots[1])]
                    constraints.append(dict(dataset=name,component=component,quadratic=[a,b,c],limit=bound,
                        positive_feasible_interval=[max(0.,roots[0]),roots[1]]))
            feasible=interval[0]<=interval[1]
            sampled=[]
            if feasible:
                for gain in np.linspace(*interval,201):
                    a=gain-1;power=np.maximum(moments['S_energy'][:,color]-2*a*moments['SP_energy'][:,color]+a*a*moments['P_energy'][:,color],0.)
                    s_gain=float(weights@np.sqrt(power/moments['S_target_energy'][:,color]))
                    if .75<=s_gain<=1.25:sampled.append(float(gain))
            colors.append(dict(color=color,analytic_NMSE_and_O_gain_interval=interval,nonempty=feasible,
                additionally_S_gain_checked_grid=sampled,constraints=constraints))
        results[variant]=dict(changed_inputs=changed[variant],colors=colors)
    write_json(out/'summary.json',dict(status='completed_diagnostic',source_checkpoint=str(source_path),
        source_ids_sha256=hashlib.sha256((audit/'integer_ids.npz').read_bytes()).hexdigest(),
        all_development_inputs=len(examples),results=results,
        interpretation='feasibility_only_no_gain_selected_from_development',confirmation_arrays_read=0,test_arrays_read=0))


def export_transfer_arms(out,saved,signature,source_path,quiet,quiet_path,arms,objective):
    for arm,gain in arms.items():
        folder=out/arm;folder.mkdir(exist_ok=True)
        metadata=dict(saved['metadata'],signature=signature,decoder_component_transfer_gain=gain,
            output_calibration=dict(objective=objective,
                source_checkpoint=str(source_path),source_sha256=signature['source_sha256'],
                integer_mapping='exact_frozen_source_S128_O256',decoder_dependency=dict(S='both_S_and_O_integer_sequences',O='O_integer_sequence'),
                sum='original_frozen_X_reconstruction_bitwise_preserved',arithmetic='float64_components_source_sum_dtype_preserved'))
        payload=dict(metadata=metadata,signature=signature,model_config=saved['model_config'],model_state=saved['model_state'],step=saved['step'])
        torch.save(payload,folder/'model.pt')
        pure_sha=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()
        base_sizes=metadata['branch_codebook_sizes'];sizes={key:value+1 for key,value in base_sizes.items()}
        for variant,fit in quiet['variants'].items():
            silent_metadata=dict(metadata,schema='silent_observation_tokenizer_v1',source_file='model.pt',source_sha256=pure_sha,
                base_codebook_size=max(base_sizes.values()),codebook_size=max(sizes.values()),base_codebook_sizes=base_sizes,
                branch_codebook_sizes=sizes,quiet_threshold_input_sd=fit['threshold_input_sd'],output_dtype='float64',
                quiet_training_fit=str(quiet_path),quiet_training_fit_sha256=signature['quiet_training_fit_sha256'],variant=variant,
                reserved_pair='S128_and_O256_repeated_for_all_fifteen_positions')
            torch.save(dict(metadata=silent_metadata,level=np.asarray(fit['level_S'],float)),folder/(variant+'.pt'))


def project_training_energy(args,cfg):
    out=args.output
    if (out/'summary.json').exists():raise FileExistsError('completed energy projection retained')
    manifest,scales,_,_=base_tokenizer_training_data(cfg,SOURCE)
    previous=SOURCE/cfg['reuse_training_fit'];record=json.loads((previous/'summary.json').read_text())
    source_path=SOURCE/cfg['source_checkpoint'];source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
    if (record['signature']['input']!=identity(manifest) or record['signature']['scales']!=identity(scales)
            or record['signature']['source_sha256']!=source_sha):raise ValueError('train projection source identity mismatch')
    previous_cfg=json.loads((previous/'resolved_config.json').read_text())
    if any(previous_cfg['fit'][key]!=cfg['fit'][key] for key in ('gain_grid','macro_nmse_ratio_max','dataset_nmse_ratio_max','rms_gain_interval')):
        raise ValueError('projection must preserve the frozen training constraint family')
    search_path=previous/'training_gain_search.json';search=json.loads(search_path.read_text());selected=[]
    for color in search:
        feasible=[value for value in color['candidates'] if value['eligible']]
        selected.append(min(feasible,key=lambda value:(abs(value['rms_gain'][1]-cfg['fit']['energy_match_target']),value['objective'])))
    gains=[value['gain'] for value in selected];arms={'HbO_only':[gains[0],1.],'both_colors':gains}
    signature=dict(config=identity(cfg),input=identity(manifest),scales=identity(scales),source_sha256=source_sha,
        training_gain_search_sha256=hashlib.sha256(search_path.read_bytes()).hexdigest(),
        training_moments_sha256=hashlib.sha256((previous/'training_transfer_moments.npz').read_bytes()).hexdigest(),
        quiet_training_fit_sha256=record['signature']['quiet_training_fit_sha256'])
    quiet_path=SOURCE/cfg['quiet_training_fit']
    if hashlib.sha256(quiet_path.read_bytes()).hexdigest()!=signature['quiet_training_fit_sha256']:raise ValueError('quiet training fit drift')
    saved=torch.load(source_path,map_location='cpu',weights_only=False);quiet=json.loads(quiet_path.read_text())
    export_transfer_arms(out,saved,signature,source_path,quiet,quiet_path,arms,
        'train_O_RMS_energy_matching_to_one_projected_onto_original_train_NMSE_and_energy_feasible_grid')
    write_json(out/'resolved_config.json',cfg)
    write_json(out/'summary.json',dict(status='completed',signature=signature,training_contexts=record['training_contexts'],
        gain=gains,arms=arms,train_selected=selected,training_constraints=cfg['fit'],source_training_fit=str(previous),
        training_arrays_reread=0,validation_arrays_read=0,test_arrays_read=0,selected_arm='HbO_only',
        selected_arm_rule='predeclared_HbO_only_primary_both_colors_diagnostic',new_training_updates=0,
        pure_integer_rates=dict(S=3.5,O=4.),quiet_integer_rates=dict(S=4.,O=4.5)))


def transfer_fit(args,cfg):
    out=args.output
    if args.synthetic:
        p=np.linspace(-1,1,600).reshape(1,15,40);s=.2*np.cos(p)
        prediction=dict(S=s,O=p,sum=s+p)
        batch=dict(s=s-.2*p,o=1.2*p,x=s+p,mask=np.ones_like(p,bool),vs=np.ones_like(p),vo=np.ones_like(p))
        moments=transfer_moments(prediction,batch)
        a=(moments['S_b']-moments['O_b'])/(moments['S_a']+moments['O_a'])
        np.testing.assert_allclose(a,.2,rtol=0,atol=1e-12)
        write_json(out/'synthetic.json',dict(status='passed',expected_O_gain=1.2,training_moments_recover_gain=a+1,source_sum_preserved=True))
        return
    if (out/'summary.json').exists():raise FileExistsError('completed paired calibration retained')
    torch.set_num_threads(cfg['resources']['torch_threads'])
    manifest,scales,_,_=base_tokenizer_training_data(cfg,SOURCE);rows=manifest['rows']
    source_path=SOURCE/cfg['source_checkpoint'];saved=torch.load(source_path,map_location='cpu',weights_only=False)
    source=load_model(source_path)
    if source.metadata['signature']['input']!=identity(manifest) or source.metadata['scales_identity']!=manifest['scales_identity']:
        raise ValueError('source data/scales identity mismatch before arrays')
    if any(source.metadata.get(key) is not None for key in ('decoder_component_transfer_gain','decoder_color_gain','decoder_lowpass_hz')):
        raise ValueError('uncalibrated paired source required')
    examples=np.asarray(base_tokenizer_examples(rows,'hb','train'),np.int32)
    if any(rows[i]['split']!='train' for i,_,_ in examples):raise ValueError('training boundary before arrays')
    quiet_path=SOURCE/cfg['quiet_training_fit'];quiet=json.loads(quiet_path.read_text())
    g0=SOURCE/cfg['data']['run']
    for key,path in [('manifest_sha256',g0/'cache/manifest.json'),('scales_sha256',g0/'scales.json')]:
        if quiet['source'][key]!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('quiet training fit identity mismatch')
    signature=dict(config=identity(cfg),input=identity(manifest),scales=identity(scales),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),quiet_training_fit_sha256=hashlib.sha256(quiet_path.read_bytes()).hexdigest())
    if (out/'signature.json').exists() and json.loads((out/'signature.json').read_text())!=signature:
        raise ValueError('partial training calibration identity mismatch before arrays')
    write_json(out/'resolved_config.json',cfg);write_json(out/'signature.json',signature)
    moments={};cache=OrderedDict()
    for start in range(0,len(examples),cfg['resources']['batch_size']):
        take=examples[start:start+cfg['resources']['batch_size']]
        batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=128)
        values=transfer_moments(source(batch['x']),batch)
        for key,value in values.items():moments.setdefault(key,[]).append(value)
        if start%1024==0:write_json(out/'progress.json',dict(stage='all_train_transfer_moments',completed=start+len(take),planned=len(examples)))
    moments={key:np.concatenate(value) for key,value in moments.items()}
    np.savez_compressed(out/'training_transfer_moments.npz',examples=examples,**moments)
    weights=training_weights(examples,rows)
    datasets=np.asarray([rows[i]['dataset'] for i,_,_ in examples])
    def aggregate(value):return np.einsum('n,nc->c',weights,value)
    coefficients={key:aggregate(value) for key,value in moments.items()}
    by_dataset={name:{key:np.einsum('n,nc->c',weights[datasets==name]/weights[datasets==name].sum(),value[datasets==name])
        for key,value in moments.items()} for name in sorted(set(datasets))}
    grids=[];details=[]
    for color in range(2):
        candidates=[]
        for gain in np.linspace(*cfg['fit']['gain_grid']):
            a=gain-1
            def nmse(coef):return np.array([coef['S_c'][color]-2*a*coef['S_b'][color]+a*a*coef['S_a'][color],
                coef['O_c'][color]+2*a*coef['O_b'][color]+a*a*coef['O_a'][color]])
            value=nmse(coefficients)
            macro_pass=bool((value<=cfg['fit']['macro_nmse_ratio_max']*np.array([coefficients['S_c'][color],coefficients['O_c'][color]])).all())
            dataset_pass=all((nmse(coef)<=cfg['fit']['dataset_nmse_ratio_max']*np.array([coef['S_c'][color],coef['O_c'][color]])).all() for coef in by_dataset.values())
            energy=np.maximum(moments['S_energy'][:,color]-2*a*moments['SP_energy'][:,color]+a*a*moments['P_energy'][:,color],0.)
            def rms_ratio(predicted,target):
                valid=target>0
                return float(np.sum(weights[valid]*np.sqrt(predicted[valid]/target[valid]))/weights[valid].sum())
            rms=np.array([rms_ratio(energy,moments['S_target_energy'][:,color]),
                gain*rms_ratio(moments['P_energy'][:,color],moments['O_target_energy'][:,color])])
            energy_pass=bool(((rms>=cfg['fit']['rms_gain_interval'][0])&(rms<=cfg['fit']['rms_gain_interval'][1])).all())
            candidates.append(dict(gain=float(gain),nmse=value,rms_gain=rms,objective=float(value.mean()),eligible=macro_pass and dataset_pass and energy_pass,
                macro_pass=macro_pass,dataset_pass=bool(dataset_pass),energy_pass=energy_pass))
        eligible=[value for value in candidates if value['eligible']]
        chosen=min(eligible,key=lambda value:(value['objective'],abs(value['gain']-1))) if eligible else None
        details.append(dict(color=color,feasible=len(eligible),selected=chosen,candidates=candidates))
        grids.append(chosen)
    write_json(out/'training_gain_search.json',details)
    if any(value is None for value in grids):
        write_json(out/'summary.json',dict(status='training_constraints_infeasible',signature=signature,colors=details,
            training_contexts=len(examples),validation_arrays_read=0,test_arrays_read=0));return
    gains=[value['gain'] for value in grids]
    arms={'HbO_only':[gains[0],1.],'both_colors':gains}
    r_unmodified=next(value for value in details[1]['candidates'] if abs(value['gain']-1)<1e-12)
    objectives={'HbO_only':(grids[0]['objective']+r_unmodified['objective'])/2,
        'both_colors':(grids[0]['objective']+grids[1]['objective'])/2}
    selected_arm='HbO_only' if r_unmodified['eligible'] and objectives['HbO_only']<=cfg['fit']['parsimony_objective_ratio']*objectives['both_colors'] else 'both_colors'
    export_transfer_arms(out,saved,signature,source_path,quiet,quiet_path,arms,
        'all_train_hierarchical_mean_S_NMSE_plus_O_NMSE_with_component_energy_and_error_constraints')
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=signature['source_sha256']:raise AssertionError('frozen source changed')
    write_json(out/'summary.json',dict(status='completed',signature=signature,training_contexts=len(examples),gain=gains,
        arms=arms,training_constraints=cfg['fit'],validation_arrays_read=0,test_arrays_read=0,
        selected_arm=selected_arm,training_objectives=objectives,HbO_only_training_eligible=r_unmodified['eligible'],
        pure_integer_rates=dict(S=3.5,O=4.),quiet_integer_rates=dict(S=4.,O=4.5),new_training_updates=0))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--development-source',type=Path)
    parser.add_argument('--development-baseline',type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['schema']=='joint_hb_component_transfer_gain_v1':
        args.output.mkdir(parents=True,exist_ok=True)
        if args.synthetic:
            transfer_fit(args,cfg)
            return
        if args.development_source is not None:
            if args.development_baseline is None:raise ValueError('matching complete development baseline required')
            development_family(args,cfg)
            return
        if cfg.get('reuse_training_fit') is not None:
            project_training_energy(args,cfg)
            return
        transfer_fit(args,cfg)
        return
    if cfg['schema'] != 'joint_hb_output_gain_v1':
        raise ValueError('versioned training-only O gain contract required')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.synthetic:
        p = np.linspace(-1,1,600).reshape(1,15,40)
        s = np.ones_like(p)*.3
        batch = dict(x=s+2*p,o=2*p,mask=np.ones_like(p,dtype=bool),vo=np.ones_like(p),vx=np.ones_like(p))
        equation = normal_equation(dict(S=s,O=p),batch,[3.])
        np.testing.assert_allclose(equation[:,1]/equation[:,0],2.,rtol=1e-12,atol=1e-12)
        write_json(args.output/'synthetic.json',dict(status='passed',expected_O_gain=2.,S_unchanged=True))
        return
    if (args.output/'summary.json').exists():
        raise FileExistsError('completed calibration retained')
    torch.set_num_threads(cfg['resources']['torch_threads'])
    manifest,scales,_,_ = base_tokenizer_training_data(cfg,SOURCE)
    rows = manifest['rows']
    panel = SOURCE/cfg['panel']
    panel_record = json.loads((panel.parent/'panel.json').read_text())
    if panel_record['panel_sha256'] != hashlib.sha256(panel.read_bytes()).hexdigest():
        raise ValueError('fixed training panel changed')
    with np.load(panel) as stored:
        examples = stored['hb_pattern_train'].copy()
    if any(rows[i]['split'] != 'train' or rows[i]['modality'] != 'hb' for i,_,_ in examples):
        raise ValueError('training-only gain fit boundary before arrays')
    source = SOURCE/cfg['source_checkpoint']
    saved = torch.load(source,map_location='cpu',weights_only=False)
    if (saved['metadata']['signature']['input'] != identity(manifest)
            or saved['metadata']['scales_identity'] != manifest['scales_identity']
            or saved['metadata'].get('decoder_color_gain') is not None):
        raise ValueError('exact uncalibrated source and fixed training scales required')
    signature = dict(config=identity(cfg),input=identity(manifest),scales=identity(scales),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_json(args.output/'resolved_config.json',cfg)
    model = load_model(source)
    cache = OrderedDict()
    equations = []
    for offset in range(0,len(examples),cfg['resources']['batch_size']):
        take = examples[offset:offset+cfg['resources']['batch_size']]
        batch = base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=128)
        prediction = model(batch['x'])
        factors = [scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][p]]['scale'] for i,_,p in take]
        equations.append(normal_equation(prediction,batch,factors))
        if offset%512==0:
            write_json(args.output/'progress.json',dict(stage='training_gain_fit',completed=offset+len(take),total=len(examples)))
    equation = np.concatenate(equations)
    aggregate = base_training_aggregate({'nmse':equation},examples.tolist(),rows)['metrics']['nmse']
    matrix = np.asarray(aggregate['macro'])
    gain = matrix[1]/matrix[0]
    if not np.isfinite(gain).all() or (gain<=0).any():
        raise ValueError('positive finite O gain is required')
    metadata = dict(saved['metadata'],signature=signature,
        decoder_color_gain=dict(S=[1.,1.],O=gain.tolist()),
        output_calibration=dict(source=str(source),source_sha256=signature['source_sha256'],
            objective='training_hierarchical_mean_O_NMSE_plus_X_NMSE_S_fixed',
            fit_coordinate='native_y_and_prediction_divided_by_native_G0_training_component_variance',
            learned_parameters='two_dimensionless_O_color_gains',input_side_information='none'))
    np.savez_compressed(args.output/'normal_equations.npz',examples=examples,equation=equation)
    payload = dict(metadata=metadata,signature=signature,model_config=saved['model_config'],
        model_state=saved['model_state'],step=saved['step'])
    temporary = args.output/'model.pending.pt'
    torch.save(payload,temporary)
    temporary.replace(args.output/'model.pt')
    if hashlib.sha256(source.read_bytes()).hexdigest() != signature['source_sha256']:
        raise AssertionError('source checkpoint changed during frozen calibration')
    write_json(args.output/'summary.json',dict(status='completed',signature=signature,gain=gain,
        normal_equation_macro=matrix,by_dataset=aggregate['by_dataset'],training_contexts=len(examples),
        encoder_and_model_state='exact_source_state',source_checkpoint=str(source),
        validation_arrays_read=0,test_arrays_read=0))


if __name__=='__main__':
    main()
