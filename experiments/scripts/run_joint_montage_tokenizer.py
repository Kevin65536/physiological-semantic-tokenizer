#!/usr/bin/env python3
"""Train hard atomic assignments with complete 30-second montage reconstruction."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

from src.analysis.spatial_tokenizer import screen, write
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_joint_eeg
from src.tokenizers.montage_joint_tokenizer import MontageJointTokenizer, load_model
from src.tokenizers.spatial_waveform_dictionary import BANDS, load_model as load_source
from src.tokenizers.spectral_waveform_dictionary import raw_features
from experiments.scripts.run_dual_component_tokenizer import identity, owner_lock


def validate(cfg):
    if (cfg['schema'] != 'dual_component_montage_joint_v1' or cfg['training']['fitting_splits'] != ['train']
            or cfg['evaluation']['role'] != 'screen' or cfg['evaluation']['confirmation'] != 'forbidden'
            or cfg['model']['continuous_decoder_inputs'] or cfg['model']['context_s'] != 30
            or cfg['model']['token_s'] != 2 or cfg['model']['sample_rate_hz'] != 200):
        raise ValueError('train-only 30s X-only one-ID atomic contract required')


def initial_model(cfg):
    source = load_source(ROOT/cfg['model']['initialization'])
    if source.codebook_size!=cfg['model']['codebook_size'] or source.metadata.get('output_projection_hz')!=[1,45]:
        raise ValueError('declared source vocabulary and frozen 1–45 Hz output coordinate required')
    metadata = deepcopy(source.metadata)
    metadata.update(top_m=cfg['model']['top_m'], temperature=cfg['model']['temperature'],
        global_analysis=cfg['model'].get('global_analysis',False),
        joint_config_identity=identity(cfg), joint_training_steps=0,
        joint_training_seed=cfg['training']['seed'],
        encoder_rule=('frozen_Wiener_learned_full_montage_analysis_dynamic_topM_hard_atomic_assignment'
                      if cfg['model'].get('global_analysis') else 'frozen_Wiener_initial_topM_learned_hard_atomic_assignment'),
        decoder_rule='complete_30s_learned_same_and_cross_branch_spatial_synthesis',
        gradient_rule='hard_forward_soft_topM_straight_through',
        initialization_checkpoint=str(ROOT/cfg['model']['initialization']))
    return source, MontageJointTokenizer(source, metadata)


def prepare(cfg, output):
    if (output/'prepare_summary.json').exists():
        if json.loads((output/'resolved_config.json').read_text())['config']!=cfg:
            raise ValueError('completed preparation configuration changed')
        return
    began = time.monotonic()
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    source, model = initial_model(cfg)
    if model.metadata['input_identity'] != manifest['cfg_identity'] or model.metadata['scales_identity'] != manifest['scales_identity']:
        raise ValueError('frozen initialization G0 identity required')
    plan = json.loads((ROOT/cfg['training']['plan']).read_text())['plan']
    for d, subjects in plan.items():
        for subject, crops in subjects.items():
            if any(rows[c['row']]['split'] != 'train' or rows[c['row']]['dataset'] != d or rows[c['row']]['subject'] != subject for c in crops):
                raise ValueError('complete train plan validated before signal arrays')
    panel = json.loads((ROOT/cfg['evaluation']['panel']).with_name('panel.json').read_text())
    with np.load(ROOT/cfg['evaluation']['panel'], allow_pickle=False) as stored:
        examples = stored['eeg_screen']
    if any(rows[i]['split'] != 'validation' or rows[i]['subject'] == panel['confirmation_subjects'][rows[i]['dataset']] for i, _, _ in examples):
        raise ValueError('development screen only before any arrays')
    write(output/'resolved_config.json',dict(config=cfg,metadata=model.metadata,training_plan=plan,
        screen_subjects=sorted(set((rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples)),
        confirmation_access=0, frozen_source_sha256=hashlib.sha256((ROOT/cfg['model']['initialization']).read_bytes()).hexdigest()))
    folder = output/'cache'
    folder.mkdir(exist_ok=True)
    cache = OrderedDict()
    records = []
    totals = {b: np.zeros((5,model.codebook_size,3)) for b in ('S','O')}
    counts = {b: np.zeros((5,model.codebook_size)) for b in ('S','O')}
    attr_values = {b: [] for b in ('S','O')}
    replay_max = 0.
    for dataset, subjects in sorted(plan.items()):
        names = model.metadata['positions'][dataset]
        refs = {n.upper():v for n,v in scales['positions']['eeg'][dataset].items()}
        variance = np.asarray([[refs[n.upper()]['target_variance'][f][0] for n in names] for f in ('x','s','o')],np.float32)
        for subject, crops in sorted(subjects.items()):
            for crop in crops:
                number=len(records)
                fields, observed, valid = base_tokenizer_joint_eeg(rows,crop,scales,names,cache)
                values=fields['x'].T.reshape(1,len(names),15,400)
                features,candidates=model.training_features(values,dataset)
                target=np.stack([fields[f].T.reshape(len(names),15,400) for f in ('x','s','o')]).astype(np.float32)
                attributes=np.stack([raw_features(target[b].reshape(-1,400),None,None,'theta_alpha_rms').reshape(len(names),15,3) for b in (1,2)]).astype(np.float32)
                item=dict(dataset=dataset,subject=subject,record=rows[crop['row']]['record'],crop=crop,
                    features=features,candidates=candidates,observed=torch.from_numpy(observed),valid=torch.from_numpy(valid),
                    target=torch.from_numpy(target),variance=torch.from_numpy(variance),attributes=torch.from_numpy(attributes))
                path=folder/f'{number:04d}.pt'
                torch.save(item,path)
                records.append(dict(dataset=dataset,subject=subject,record=item['record'],crop=crop,path=str(path)))
                roles=model.roles(dataset)
                for b,branch in enumerate(('S','O')):
                    attr_values[branch].append(attributes[b,valid].reshape(-1,3))
                    for band in range(5):
                        p=np.flatnonzero(roles==band)
                        codes=candidates[f'{branch}_{band}'][:,0].numpy().reshape(len(p),15)+1
                        keep=np.broadcast_to(valid[p,None],codes.shape)
                        np.add.at(totals[branch][band],codes[keep],attributes[b,p][keep])
                        np.add.at(counts[branch][band],codes[keep],1)
                if number < 4:
                    ids=model.encode(values,dataset,observed[None])
                    frozen=source.decode_ids(ids,dataset)
                    joint=model.decode_ids(ids,dataset)
                    for branch in ('S','O','sum'):
                        difference=float(np.max(abs(frozen[branch]-joint[branch])))
                        replay_max=max(replay_max,difference)
                        np.testing.assert_allclose(joint[branch],frozen[branch],atol=3e-6,rtol=2e-5)
                if number % 20 == 0:
                    write(output/'progress.json',dict(stage='prepare',crops=len(records),elapsed_seconds=time.monotonic()-began))
    stats={}
    for branch in ('S','O'):
        attr=np.concatenate(attr_values[branch])
        mean,scale=attr.mean(0),attr.std(0).clip(.1)
        stats[branch]=dict(mean=mean,scale=scale)
        with torch.no_grad():
            for band in range(5):
                avg=totals[branch][band]/counts[branch][band,:,None].clip(1)
                avg[counts[branch][band]==0]=mean
                model.network.attributes[f'{branch}_{band}'].copy_(torch.tensor((avg-mean)/scale,dtype=torch.float32))
    write(output/'training_index.json',dict(records=records,attribute_coordinate=stats,
        source_initialization_replay_max_abs=replay_max,training_crops=len(records),fitting_split='train'))
    model.save(output/'initialization.npz')
    shutil.copy2(__file__,output/'source_execution.py')
    shutil.copy2(ROOT/'src/tokenizers/montage_joint_tokenizer.py',output/'model_source.py')
    write(output/'prepare_summary.json',dict(status='completed',training_crops=len(records),
        training_subjects=len(set((i['dataset'],i['subject']) for i in records)),source_replay_max_abs=replay_max,
        elapsed_seconds=time.monotonic()-began,validation_fits=0,confirmation_reads=0))


def log_attributes(wave):
    window=torch.hann_window(400,periodic=False,dtype=wave.dtype,device=wave.device)
    centered=wave-wave.mean(-1,keepdim=True)
    spectrum=torch.fft.rfft(centered*window,dim=-1)
    density=abs(spectrum)**2*2/(400*window.square().sum())
    power=[]
    for low,high in ((4,8),(8,13)):
        power.append(torch.log(density[...,2*low:2*high].sum(-1).clamp_min(1e-12)))
    rms=torch.sqrt(torch.mean(wave.reshape(*wave.shape[:-1],8,50)**2,dim=-1)+1e-12).mean(-1)
    return torch.stack((*power,torch.log(rms.clamp_min(1e-12))),dim=-1)


def loss(network,item,stats,cfg,reference):
    atoms,attr,ids=network.assign(item['features'],item['candidates'],item['observed'],item['dataset'])
    output=network.decode(atoms,item['dataset'],observed=item['observed'])
    prediction=torch.stack((output.sum(0),output[0],output[1]))
    valid=item['valid']
    if not bool(valid.any()):
        raise ValueError('empty training target support')
    wave=((prediction[:,valid]-item['target'][:,valid])**2/item['variance'][:,valid,None,None]).mean()
    actual=log_attributes(output[:,valid])
    target=item['attributes'][:,valid]
    energy=((actual-target)**2).mean()
    normalized=torch.stack([(item['attributes'][b]-stats[branch][0])/stats[branch][1] for b,branch in enumerate(('S','O'))])
    semantic=((attr[:,valid]-normalized[:,valid])**2).mean()
    regularization=sum((p-reference[n]).square().mean() for n,p in network.named_parameters() if p.requires_grad and not n.startswith('attributes'))
    objective=wave+cfg['training']['output_log_energy_weight']*energy+cfg['training']['single_ID_attribute_weight']*semantic+cfg['training']['initialization_l2_weight']*regularization
    return objective,dict(wave=float(wave.detach()),log_energy=float(energy.detach()),single_ID=float(semantic.detach()),total=float(objective.detach()))


def train(cfg,output,arm):
    began=time.monotonic()
    if (output/arm/'training_summary.json').exists():
        if json.loads((output/'resolved_config.json').read_text())['config']!=cfg:
            raise ValueError('completed training configuration changed')
        return
    index=json.loads((output/'training_index.json').read_text())
    records=index['records']
    data=[torch.load(r['path'],weights_only=False,map_location='cpu') for r in records]
    eligible=[n for n,item in enumerate(data) if bool(item['valid'].any())]
    if not eligible:
        raise ValueError('no complete train target positions in retained plan')
    coordinate={b:(torch.tensor(index['attribute_coordinate'][b]['mean']),torch.tensor(index['attribute_coordinate'][b]['scale'])) for b in ('S','O')}
    target=output/arm
    target.mkdir(exist_ok=True)
    model=load_model(output/'initialization.npz')
    if model.metadata['joint_config_identity']!=identity(cfg):
        raise ValueError('retained initialization configuration changed')
    model.metadata.update(arm=arm)
    network=model.network
    network.train()
    groups=[]
    for name,parameters,lr in [('atoms',network.atoms,cfg['training']['learning_rate_atoms']),
        ('analysis',network.analysis,cfg['training'].get('learning_rate_analysis',.0001)),
        ('centers',network.centers,cfg['training']['learning_rate_assignment']),('bias',network.bias,cfg['training']['learning_rate_assignment']),
        ('synthesis',network.synthesis,cfg['training']['learning_rate_synthesis']),('attributes',network.attributes,cfg['training']['learning_rate_attributes'])]:
        active=arm=='joint' or name in ('synthesis','attributes')
        for p in parameters.values():p.requires_grad_(active)
        if active: groups.append(dict(params=list(parameters.values()),lr=lr))
    optimizer=torch.optim.Adam(groups)
    reference={n:p.detach().clone() for n,p in network.named_parameters()}
    nested=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for n in eligible:
        r=records[n]
        nested[r['dataset']][r['subject']][r['record']].append(n)
    write(target/'training_support.json',dict(planned_crops=len(records),eligible_crops=len(eligible),
        no_complete_target_crops=[records[n] for n in range(len(records)) if n not in eligible],
        rule='only complete G0 target positions enter loss; empty-target planned crops retained and excluded from sampling'))
    rng=np.random.default_rng(cfg['training']['seed'])
    def choose(values):return values[int(rng.integers(len(values)))]
    steps=cfg['training']['steps']
    schedule=[]
    for _ in range(steps):
        d=choose(sorted(nested));s=choose(sorted(nested[d]));r=choose(sorted(nested[d][s]));schedule.append(choose(nested[d][s][r]))
    checkpoint=target/'training_state.pt'
    first=0
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False,map_location='cpu')
        network.load_state_dict(state['network']);optimizer.load_state_dict(state['optimizer']);first=state['step']
    if first==0 and not (target/'checkpoint_step_0000.npz').exists():
        model.save(target/'checkpoint_step_0000.npz')
    for step in range(first+1,steps+1):
        item=data[schedule[step-1]]
        optimizer.zero_grad(set_to_none=True)
        objective,metrics=loss(network,item,coordinate,cfg,reference)
        if not torch.isfinite(objective):raise FloatingPointError('nonfinite joint objective')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),cfg['training']['gradient_clip_norm'])
        optimizer.step()
        with torch.no_grad():
            for p in network.atoms.values():p[0].zero_()
        if step % 10==0 or step==1:
            record=dict(step=step,arm=arm,elapsed_seconds=time.monotonic()-began,**metrics)
            with (target/'training_log.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            write(target/'progress.json',dict(status='training',planned_steps=steps,estimated_remaining_seconds=(time.monotonic()-began)/max(step-first,1)*(steps-step),**record))
            print(json.dumps(record),flush=True)
        if step in cfg['training']['milestones']:
            model.metadata['joint_training_steps']=step
            model.save(target/f'checkpoint_step_{step:04d}.npz')
            pending=checkpoint.with_suffix('.pending.pt')
            torch.save(dict(step=step,network=network.state_dict(),optimizer=optimizer.state_dict()),pending);pending.replace(checkpoint)
    model.save(target/'model.npz')
    write(target/'training_summary.json',dict(status='completed',arm=arm,steps=steps,elapsed_seconds=time.monotonic()-began,
        training_crops=len(records),matched_schedule_sha256=hashlib.sha256(np.asarray(schedule).tobytes()).hexdigest(),
        fitted_groups=[n for n,p in network.named_parameters() if p.requires_grad],validation_fits=0,confirmation_reads=0))


def evaluate(cfg,output,arm,step):
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    target=output/arm
    checkpoint=target/f'checkpoint_step_{step:04d}.npz'
    model=load_model(checkpoint)
    if model.metadata['joint_config_identity']!=identity(cfg):
        raise ValueError('frozen checkpoint configuration changed')
    audit=target/f'screen_step_{step:04d}'
    audit.mkdir(exist_ok=True)
    if (audit/'screen_summary.json').exists():return
    summaries=screen(model,checkpoint,manifest['rows'],scales,cfg,audit,mode_names=('spatial',))
    write(audit/'summary.json',dict(status='completed',arm=arm,step=step,scores={k:v['selection_score'] for k,v in summaries.items()}))


def audit(cfg,output,arm,step):
    from argparse import Namespace
    from experiments.scripts import audit_spatial_tokenizer_patterns as pattern
    checkpoint=output/arm/f'checkpoint_step_{step:04d}.npz'
    if load_model(checkpoint).metadata['joint_config_identity']!=identity(cfg):
        raise ValueError('frozen single-ID audit configuration changed')
    target=output/arm/f'screen_step_{step:04d}'
    pattern_output=target/'pattern'
    args=Namespace(model=checkpoint,panels=ROOT/cfg['evaluation']['panel'],output=pattern_output,batch_size=64)
    if not (pattern_output/'support_summary.json').exists():pattern.run(args)
    if not (pattern_output/'attribute_summary.json').exists():pattern.assess_attributes(args)
    export_standard(cfg,checkpoint,target)
    if (pattern_output/'training_attribute_profiles.npz').exists() and not (pattern_output/'major_atom_atlas.png').exists():
        atom_atlas(checkpoint,pattern_output,cfg)
    if not (target/'integer_rate.json').exists():integer_rate(checkpoint,target)


def export_standard(cfg,checkpoint,target):
    from argparse import Namespace
    from experiments.scripts.evaluate_joint_tokenizer import load_source as load_scope,display_selection,input_scales
    from experiments.scripts.render_joint_tokenizer_waveforms import render_export
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    export=target/'standard_export_v1'
    export.mkdir(exist_ok=True)
    args=Namespace(panel=ROOT/cfg['evaluation']['panel'],modality='eeg',role='screen',
        reference_metrics=ROOT/cfg['evaluation']['panel'].rsplit('/',1)[0]/'baseline_audit/eeg/vq128/metrics.npz')
    if not (export/'waveform_examples.json').exists():
        rows,scales,examples,manifest=load_scope(args)
        with np.load(target/'spatial_metrics.npz') as stored:
            np.testing.assert_array_equal(stored['examples'],examples)
            metrics={k:stored[k].copy() for k in stored if k!='examples' and not k.startswith('support_')}
        selected=display_selection(args,metrics,rows,examples)
        indices=sorted(selected);take=examples[indices]
        batch=base_tokenizer_batch(rows,take,scales)
        model=load_model(checkpoint)
        packets=json.loads((target/'integer_metadata.json').read_text())
        crops={(i['row'],i['start_s']):i for i in packets}
        prediction={k:np.empty_like(batch['x'],dtype=np.float64) for k in ('S','O','sum')}
        replayed={}
        with np.load(target/'integer_ids.npz') as stored:
            for n,(i,start,p) in enumerate(take):
                row=rows[i];packet=crops[int(i),int(start)];number=packet['crop']
                if number not in replayed:
                    replayed[number]=model.decode_ids({b:stored[f'{b}_{number}'].copy() for b in ('S','O')},row['dataset'])
                native=model.metadata['positions'][row['dataset']]
                column=next(j for j,name in enumerate(native) if name.upper()==row['positions'][p].upper())
                for k in prediction:prediction[k][n]=replayed[number][k][0,column]
        arrays={k:batch[k] for k in ('x','s','o','mask')}
        arrays.update({'prediction_'+k:v for k,v in prediction.items()})
        np.savez_compressed(export/'waveform_examples.npz',**arrays)
        records=[]
        for index in indices:
            i,start,p=examples[index];row=rows[i]
            records.append(dict(modality='eeg',dataset=row['dataset'],subject=row['subject'],record=row['record'],parent=row['parent'],
                position_name=row['positions'][p],row=int(i),position=int(p),start_s=int(start),
                scale=float(input_scales(rows,[examples[index]],scales)[0]),source_unit=row['source_unit'],selection=selected[index],
                panel_index=index,nmse=metrics['nmse'][index].tolist(),token_seconds=2,prediction_label='Joint integer decode'))
        write(export/'waveform_examples.json',dict(coordinate='G0_normalized',model=str(checkpoint),role='screen',
            reference_metrics=str(args.reference_metrics),examples=records,
            provenance=dict(integer_source=str(target/'integer_ids.npz'),selection='existing frozen display_selection rules',
                prediction='independent complete native integer packet decode then selected physical position')))
    if not (target/'standard_figures_v1').exists():
        render_export(export/'waveform_examples.npz',export/'waveform_examples.json',target/'standard_figures_v1')


def integer_rate(checkpoint,output):
    model=load_model(checkpoint)
    metadata=json.loads((output/'integer_metadata.json').read_text())
    summary=json.loads((output/'screen_summary.json').read_text())
    counts={(b,k):np.zeros(model.codebook_size,np.int64) for b in ('S','O') for k in range(5)}
    total=0;position_seconds=0
    with np.load(output/'integer_ids.npz') as stored:
        for packet in metadata:
            names=model.metadata['positions'][packet['dataset']]
            if names!=packet['native_positions']:raise ValueError('integer packet native order changed')
            roles=model.roles(packet['dataset']);position_seconds+=30*len(names)
            for b in ('S','O'):
                codes=stored[f'{b}_{packet["crop"]}']
                if codes.shape!=(1,len(names),15) or codes.dtype.kind not in 'iu':raise ValueError('single-ID integer packet required')
                total+=codes.size
                for k in range(5):counts[b,k]+=np.bincount(codes[:,roles==k].ravel(),minlength=model.codebook_size)
    namespaces=[];entropy_total=0.
    for (b,k),value in counts.items():
        p=value[value>0]/max(value.sum(),1);entropy=float(-np.sum(p*np.log2(p)));entropy_total+=entropy*value.sum()
        namespaces.append(dict(branch=b,role_band_hz=BANDS[k],all_integer_slots=int(value.sum()),
            missing_ID0_slots=int(value[0]),occupied_nonmissing_codes=int(np.count_nonzero(value[1:])),conditional_entropy_bits_per_ID=entropy))
    width=int(np.ceil(np.log2(model.codebook_size)));storage=(output/'integer_ids.npz').stat().st_size*8
    write(output/'integer_rate.json',dict(status='completed',namespace='branch/fixed_role_band/integer_ID',
        integers_per_2s_per_branch_per_native_position=1,codebook_size_including_missing=model.codebook_size,
        integer_slots=total,full_native_position_seconds=position_seconds,fixed_width_bits_per_ID=width,
        nominal_fixed_width_total_bits=total*width,nominal_bits_per_native_position_second_both_branches=total*width/position_seconds,
        empirical_namespace_entropy_total_bits=entropy_total,entropy_bits_per_native_position_second=entropy_total/position_seconds,
        serialized_NPZ_bits=storage,serialized_bits_per_native_position_second=storage/position_seconds,
        selected_scoring_position_seconds=summary['planned_examples']*30,
        nominal_bits_per_selected_scoring_position_second=total*width/(summary['planned_examples']*30),
        storage='int32 NPZ with ZIP compression; fixed-width and empirical entropy are rate accounting, not implemented bit packing',
        includes_all_auxiliary_native_electrode_integers=True,decoder_continuous_inputs=0,by_namespace=namespaces))


def atom_atlas(checkpoint,output,cfg):
    """Show one train-mass-ranked atom per namespace with a real screen member."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    model=load_model(checkpoint)
    with np.load(output/'training_attribute_profiles.npz') as stored:
        profiles={k:stored[k].copy() for k in stored.files}
    with np.load(output/'validation_IDs.npz') as stored:
        examples=stored['examples'].copy();bands=stored['fixed_band'].copy()
        ids={b:stored[b].copy() for b in ('S','O')}
    fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    records=[];cache=OrderedDict()
    time_axis=np.arange(400)/200
    for column,branch in enumerate(('S','O')):
        with (output/f'{branch}_code_support.csv').open() as stream:
            support={(int(v['fixed_band_index']),int(v['ID'])):v for v in csv.DictReader(stream)}
        for band in range(5):
            counts=profiles[branch+'_train_count'][band]
            code=int(np.argmax(counts[1:])+1)
            atom=model.network.atoms[f'{branch}_{band}'][code].detach().numpy()
            ax=axes[band,column]
            ax.plot(time_axis,atom,color='#ca4f25',lw=1,label='Fixed atomic carrier')
            candidates=np.argwhere((ids[branch]==code)&(bands[:,None]==band))
            member=None
            for row_index,patch in candidates[:64]:
                batch=base_tokenizer_batch(manifest['rows'],[examples[row_index]],scales,array_cache=cache,cache_limit=8)
                if not batch['mask'][0,patch].all():continue
                wave=batch[branch.lower()][0,patch]
                ax.plot(time_axis,wave,color='#243d56',lw=.8,alpha=.85,label='One true screen member')
                i,start,p=map(int,examples[row_index]);row=manifest['rows'][i]
                member=dict(dataset=row['dataset'],subject=row['subject'],record=row['record'],position=row['positions'][p],
                            row=i,start_s=start,patch=int(patch),absolute_patch_start_s=start+2*int(patch))
                break
            attr=profiles[branch+'_train_attribute_mean'][band,code]
            subjects=int(support[band,code]['train_subjects'])
            ax.set_title(f'{branch} | carrier role {BANDS[band][0]}–{BANDS[band][1]} Hz | ID {code}\n'
                         f'{int(counts[code])} complete train patches / {subjects} train subjects\n'
                         f'Train mean θ power {attr[0]:.3g}, α power {attr[1]:.3g}, quarter-s RMS {attr[2:].mean():.3g}',fontsize=9)
            ax.set_xlabel('Time within one 2 s patch (s)');ax.set_ylabel('G0 input SD');ax.grid(alpha=.15)
            if band==0:ax.legend(fontsize=8)
            records.append(dict(branch=branch,role_band_hz=BANDS[band],ID=code,train_complete_patches=int(counts[code]),
                train_subjects=subjects,train_theta_power=float(attr[0]),train_alpha_power=float(attr[1]),
                train_mean_quarter_second_RMS=float(attr[2:].mean()),screen_member=member,
                prototype='learned 400-sample atomic carrier before full-montage synthesis'))
    fig.suptitle('Train-mass-ranked single atomic IDs and true held-out members\nOne most frequent nonmissing ID per branch and carrier-role namespace; axes scale independently by panel',fontsize=12)
    fig.supxlabel('Attribute means use complete true train patches. Power units are squared G0 input SD.\n'
                  'The atomic carrier is one decoder input; final reconstruction uses the complete native-montage integer packet.',fontsize=10)
    fig.savefig(output/'major_atom_atlas.png',dpi=220);plt.close(fig)
    write(output/'major_atom_atlas.json',dict(status='completed',selection='maximum_complete_train_patch_count_nonmissing_ID_per_branch_role',
        full_code_profiles='training_attribute_profiles.npz',complete_code_support='S_code_support.csv and O_code_support.csv',examples=records))


def summarize(cfg,output):
    from experiments.scripts.review_joint_tokenizer import score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if (output/'summary.json').exists():return
    step=cfg['training']['steps'];arms=('joint','decoder_control')
    final={a:json.loads((output/a/f'screen_step_{step:04d}/screen_summary.json').read_text()) for a in arms}
    metrics={a:record['summaries']['spatial']['metrics'] for a,record in final.items()}
    training={a:json.loads((output/a/'training_summary.json').read_text()) for a in arms}
    if training['joint']['matched_schedule_sha256']!=training['decoder_control']['matched_schedule_sha256']:
        raise ValueError('matched sampling schedule changed')
    contract=yaml.safe_load((ROOT/cfg['evaluation']['acceptance']).read_text())
    root=ROOT/contract['run']
    baseline=json.loads((root/'baseline_audit/eeg/vq128/summary_v2.json').read_text())
    continuous=json.loads((root/'baseline_audit/eeg/continuous/summary_v2.json').read_text())
    gates={};attributes={};support={};rates={};trajectory={}
    for arm in arms:
        folder=output/arm/f'screen_step_{step:04d}'
        candidate=deepcopy(final[arm]['summaries']['spatial'])
        candidate.update(modality='eeg',role='screen',checkpoint=final[arm]['checkpoint'],
            integer_replay=dict(status='passed' if final[arm]['integer_replay_max_abs']==0 and final[arm]['standalone_metric_replay'].startswith('exact') else 'failed'))
        gates[arm]=score(candidate,baseline,continuous,contract)
        if not (folder/'reconstruction_acceptance.json').exists():write(folder/'reconstruction_acceptance.json',gates[arm])
        attributes[arm]=json.loads((folder/'pattern/attribute_summary.json').read_text())['branches']
        support[arm]=json.loads((folder/'pattern/support_summary.json').read_text())['support']
        rates[arm]=json.loads((folder/'integer_rate.json').read_text())
        trajectory[arm]={}
        for milestone in cfg['training']['milestones']:
            source=output/('joint' if milestone==0 else arm)/f'screen_step_{milestone:04d}/screen_summary.json'
            record=json.loads(source.read_text())
            trajectory[arm][str(milestone)]={k:record['summaries']['spatial']['metrics'][k]['macro'] for k in ('nmse','rms_gain','correlation')}
    left=metrics['joint']['nmse']['by_subject'];right=metrics['decoder_control']['nmse']['by_subject'];groups={}
    for a,b in zip(left,right):
        if (a['dataset'],a['subject'])!=(b['dataset'],b['subject']):raise ValueError('paired screen subjects changed')
        groups.setdefault(a['dataset'],[]).append(np.asarray(a['value']).ravel()-np.asarray(b['value']).ravel())
    rng=np.random.default_rng(cfg['training']['seed']);boots=[]
    for _ in range(5000):
        boots.append(np.mean([np.asarray(v)[rng.integers(len(v),size=len(v))].mean(0) for v in groups.values()],axis=0))
    paired=dict(joint_minus_control=np.asarray(metrics['joint']['nmse']['macro']).ravel()-np.asarray(metrics['decoder_control']['nmse']['macro']).ravel(),
        bootstrap_95_percentile=np.quantile(boots,[.025,.975],axis=0),bootstrap_unit='subject_resampled_within_dataset_then_equal_dataset_mean',
        draws=5000,multiplicity='none_exploratory_frozen_model_intervals',
        improved_subject_fraction=np.mean([np.asarray(a['value'])<np.asarray(b['value']) for a,b in zip(left,right)],axis=0))
    changed={}
    with np.load(output/'joint/screen_step_0000/integer_ids.npz') as old,np.load(output/'joint'/f'screen_step_{step:04d}/integer_ids.npz') as new:
        for branch in ('S','O'):
            keys=[k for k in old if k.startswith(branch+'_')]
            total=sum(int(np.count_nonzero(old[k])) for k in keys);difference=sum(int(np.sum(old[k]!=new[k])) for k in keys)
            changed[branch]=dict(nonmissing=total,changed=difference,fraction=difference/max(total,1))
    milestones=cfg['training']['milestones'];colors={'joint':'#ca4f25','decoder_control':'#287b9b'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for component,ax in enumerate(axes):
        for arm in arms:ax.plot(milestones,[trajectory[arm][str(n)]['nmse'][component][0] for n in milestones],marker='o',color=colors[arm],label=arm)
        ax.axhline(gates['joint']['checks']['nmse']['limits'][component][0],color='black',ls=':',label='original acceptance limit')
        ax.set(title=f'{("X","S","O")[component]} waveform NMSE',xlabel='Matched training steps',ylabel='G0 training variance units')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Full-montage hard atomic tokenizer | fixed screen: 16 subjects, four datasets\nFinal {step}-step checkpoint remains the declared primary; all milestones retained',fontsize=12)
    fig.savefig(output/'comparison_curves.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for offset,arm in [(-.18,'decoder_control'),(.18,'joint')]:
        axes[0].bar(np.arange(3)+offset,np.asarray(metrics[arm]['rms_gain']['macro']).ravel(),.36,color=colors[arm],label=arm)
        energy=np.r_[np.asarray(metrics[arm]['band_4_8_abs_db']['macro']).ravel(),np.asarray(metrics[arm]['band_8_13_abs_db']['macro']).ravel()]
        axes[1].bar(np.arange(6)+offset,energy,.36,color=colors[arm],label=arm)
        values=[attributes[arm][b]['gain'][key] for b in ('S','O') for key in ('power_4_8','power_8_13')]
        axes[2].bar(np.arange(4)+offset,values,.36,color=colors[arm],label=arm)
    axes[0].set(xticks=np.arange(3),xticklabels=['X','S','O'],title='RMS gain',ylim=(0,1.3));axes[0].axhspan(.75,1.25,color='green',alpha=.10)
    axes[1].set(xticks=np.arange(6),xticklabels=['X θ','S θ','O θ','X α','S α','O α'],title='Mean absolute band-energy error (dB)');axes[1].axhline(3.0103,color='black',ls=':')
    axes[2].set(xticks=np.arange(4),xticklabels=['S θ','S α','O θ','O α'],title='Single-ID gain beyond local background');axes[2].axhline(0,color='black',lw=.7)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Same frozen K{cfg["model"]["codebook_size"]} model and same complete native-montage integer packets\n{step} matched steps | true train attributes fit the readout; screen targets are used only for scoring',fontsize=12)
    fig.savefig(output/'semantic_energy_summary.png',dpi=220);plt.close(fig)
    fit_support=json.loads((output/'joint/training_support.json').read_text())
    index=json.loads((output/'training_index.json').read_text())
    excluded={r['path'] for r in fit_support['no_complete_target_crops']}
    fit_subjects=len({(r['dataset'],r['subject']) for r in index['records'] if r['path'] not in excluded})
    best_step=min(milestones,key=lambda n:np.asarray(trajectory['joint'][str(n)]['nmse']).mean())
    energy_pass=all(gates['joint']['checks'][f'band_{lo}_{hi}_{suffix}']['passed'] for lo,hi in ((4,8),(8,13)) for suffix in ('db','abs_db'))
    pattern_pass=all(v['primary_numeric_pass'] for v in attributes['joint'].values())
    support_pass=all(next(v['branch_validation_mass_on_8patch_2subject_codes'] for v in support['joint'] if v['branch']==b)>=.9 for b in ('S','O'))
    findings=[]
    if energy_pass:findings.append('theta/alpha macro signed and absolute dB gates pass for X S O')
    if pattern_pass:findings.append('both branch single-ID theta/alpha primary numeric gains pass')
    if support_pass:findings.append('both branch support-mass gates pass')
    verdict='not_accepted' if gates['joint']['failed'] or not pattern_pass or not support_pass else 'development_gates_pass_confirmation_unresolved'
    write(output/'summary.json',dict(status='completed',experiment_id=cfg['experiment_id'],component_order=['X','S','O'],role='screen',
        independent_screen_subjects=16,scoring_position_windows=final['joint']['planned_examples'],joint_native_crops=final['joint']['jointly_encoded_crops'],
        primary_step=step,trajectory=trajectory,final_macro={a:{k:metrics[a][k]['macro'] for k in ('nmse','rms_gain','correlation','local_rms_correlation','band_4_8_db','band_4_8_abs_db','band_8_13_db','band_8_13_abs_db')} for a in arms},
        paired_NMSE=paired,pattern_attributes=attributes,pattern_support={a:{b:next(v['branch_validation_mass_on_8patch_2subject_codes'] for v in support[a] if v['branch']==b) for b in ('S','O')} for a in arms},
        reconstruction_failed_checks={a:gates[a]['failed'] for a in arms},integer_replay_max_abs={a:final[a]['integer_replay_max_abs'] for a in arms},
        changed_IDs_from_frozen_initialization=changed,rate_evidence={a:str(output/a/f'screen_step_{step:04d}/integer_rate.json') for a in arms},
        fixed_model_bytes={a:(output/a/f'checkpoint_step_{step:04d}.npz').stat().st_size for a in arms},
        training_seconds={a:training[a]['elapsed_seconds'] for a in arms},matched_schedule_sha256=training['joint']['matched_schedule_sha256'],
        training_support_owner='joint/training_support.json',training_population=f'{fit_support["planned_crops"]} planned crops; {fit_support["eligible_crops"]} complete-target crops from {fit_subjects} subjects used by both matching samplers; all exclusions retained',
        verdict=verdict,supported_findings=findings,blocking_reconstruction_checks=gates['joint']['failed'],
        visual_evidence=f'joint/screen_step_{step:04d}/visual_audit.json',minimum_recorded_screen_macro_NMSE_step=best_step,
        stopping_interpretation=(f'screen mean XSO NMSE worsens after {best_step} steps; additional steps on this panel are not supported' if best_step<step else 'lowest recorded screen mean XSO NMSE occurs at the final primary budget'),
        unexecuted_acceptance_steps=['natural_overlapping_context_stability','independent_fit_replication','confirmation'],test_reads=0,confirmation_reads=0))


def phase_readout(cfg,output):
    """Frozen-integer carrier check with exact masked complex readout fitting."""
    from src.analysis.dual_component_tokenizer import base_training_aggregate
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    from src.tokenizers.montage_joint_tokenizer import phase_basis_statistics,phase_joint_quadratic,phase_coefficient,fit_phase_quadratic
    from experiments.scripts.review_joint_tokenizer import score
    if (cfg['schema']!='dual_component_montage_phase_readout_v1' or cfg['training']['fitting_splits']!=['train']
            or cfg['evaluation']['role']!='screen' or cfg['evaluation']['confirmation']!='forbidden'):
        raise ValueError('train-only frozen-integer phase check required')
    if (output/'summary.json').exists():return
    began=time.monotonic();manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT);rows=manifest['rows']
    checkpoint=ROOT/cfg['source']['checkpoint'];base=load_model(checkpoint)
    if base.metadata['joint_training_steps']!=6000 or base.phase_readout:
        raise ValueError('original unmodified 6000-step frozen checkpoint required')
    plan=json.loads((ROOT/cfg['source']['training_plan']).read_text())
    if len(plan)!=62 or any(rows[v['row']]['split']!='train' or rows[v['row']]['modality']!='eeg' for v in plan):
        raise ValueError('exact retained 62 train physical crop plan before arrays')
    if base.metadata['input_identity']!=manifest['cfg_identity'] or base.metadata['scales_identity']!=manifest['scales_identity']:
        raise ValueError('phase source G0 coordinates changed')
    resolved=dict(config=cfg,source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        plan=plan,decoder_input='original saved integers only; no observed phase',fit_mask='exact G0 valid time points')
    if (output/'resolved_config.json').exists():
        if json.loads((output/'resolved_config.json').read_text())!=resolved:raise ValueError('retained phase contract changed')
    else:write(output/'resolved_config.json',resolved)
    bins=cfg['training']['bin_hz'];folder=output/'training_statistics';folder.mkdir(exist_ok=True)
    paths=[];cache=OrderedDict()
    for ordinal,crop in enumerate(plan):
        path=folder/f'crop_{ordinal:03d}.npz';paths.append(path)
        if path.exists():continue
        row=rows[crop['row']];dataset=row['dataset'];names=base.metadata['positions'][dataset]
        fields,observed,_=base_tokenizer_joint_eeg(rows,crop,scales,names,cache,allowed_splits=('train',))
        ids=base.encode(fields['x'].T.reshape(1,len(names),15,400),dataset,observed[None])
        prediction=base.decode_ids(ids,dataset)
        examples=[(crop['row'],crop['start_s'],p) for p in crop['positions']]
        batch=base_tokenizer_batch(rows,examples,scales,array_cache=cache,cache_limit=8)
        native=[next(j for j,n in enumerate(names) if n.upper()==row['positions'][p].upper()) for p in crop['positions']]
        values={k:[] for k in ('gram','cross','constant','nmse')}
        for j,p in enumerate(native):
            wave=np.stack([prediction[b][0,p].reshape(-1) for b in ('S','O')])
            truth=np.stack([batch[k][j].reshape(-1) for k in ('x','s','o')]);mask=batch['mask'][j].reshape(-1)
            variance=np.asarray([batch['v'+k][j,0,0] for k in ('x','s','o')])
            gram,cross,energy=phase_basis_statistics(wave,truth,mask,bins)
            matrix,linear,constant=phase_joint_quadratic(gram,cross,energy,variance)
            values['gram'].append(matrix);values['cross'].append(linear);values['constant'].append(constant)
            estimate=np.stack((wave.sum(0),wave[0],wave[1]));values['nmse'].append(np.mean((estimate[:,mask]-truth[:,mask])**2,axis=1)/variance)
        np.savez_compressed(path,examples=np.asarray(examples),S=ids['S'],O=ids['O'],**{k:np.asarray(v) for k,v in values.items()})
        write(output/'progress.json',dict(stage='train_exact_phase_statistics',crops=ordinal+1,planned=62,elapsed_seconds=time.monotonic()-began))
    values={k:[] for k in ('examples','gram','cross','constant','nmse')}
    for path in paths:
        with np.load(path) as stored:
            for key in values:values[key].append(stored[key].copy())
    arrays={k:np.concatenate(v) for k,v in values.items()}
    aggregated=base_training_aggregate({k:arrays[k] for k in ('gram','cross','constant','nmse')},arrays['examples'],rows)
    subjects={}
    for key in ('gram','cross','constant'):
        for item in aggregated['metrics'][key]['by_subject']:
            subjects.setdefault((item['dataset'],item['subject']),{})[key]=np.asarray(item['value'])
    coefficients={};fitting={}
    with threadpool_limits(1):
        for dataset in base.metadata['positions']:
            selected=sorted([key for key in subjects if key[0]==dataset],key=lambda k:hashlib.sha256(str(k).encode()).hexdigest())
            candidates=[]
            for ridge in cfg['training']['ridge_fractions']:
                for smooth in cfg['training']['smoothness_fractions']:
                    held_scores=[]
                    for fold in range(4):
                        fit=[subjects[k] for n,k in enumerate(selected) if n%4!=fold];held=[subjects[k] for n,k in enumerate(selected) if n%4==fold]
                        if not fit or not held:continue
                        gram=np.mean([v['gram'] for v in fit],axis=0);linear=np.mean([v['cross'] for v in fit],axis=0)
                        vector=fit_phase_quadratic(gram,linear,bins,ridge,smooth)
                        held_scores.append(float(np.mean([vector@v['gram']@vector-2*vector@v['cross']+v['constant'] for v in held])))
                    candidates.append(dict(ridge=ridge,smoothness=smooth,fold_objectives=held_scores,held_objective=float(np.mean(held_scores))))
            chosen=min(candidates,key=lambda v:(v['held_objective'],-v['ridge'],-v['smoothness']))
            gram=np.mean([subjects[k]['gram'] for k in selected],axis=0);linear=np.mean([subjects[k]['cross'] for k in selected],axis=0)
            vector=fit_phase_quadratic(gram,linear,bins,chosen['ridge'],chosen['smoothness']);coefficients[dataset]=phase_coefficient(vector,bins)
            fitting[dataset]=dict(subjects=[k[1] for k in selected],candidates=candidates,selected=chosen,
                training_objective=float(vector@gram@vector-2*vector@linear+np.mean([subjects[k]['constant'] for k in selected])))
    frozen=dict(status='frozen_before_screen',fitting=fitting,
        source='exact masked full-30s integer-decoded real quadratics',independent_train_subjects=len(subjects),
        free_real_scalars=4*8*len(np.arange(1,45,bins)),validation_fits=0)
    if (output/'frozen_readout_fit.json').exists():
        old=json.loads((output/'frozen_readout_fit.json').read_text())
        if any(old['fitting'][d]['selected']!=fitting[d]['selected'] for d in fitting):raise ValueError('frozen phase CV choice changed')
    else:write(output/'frozen_readout_fit.json',frozen)
    contract=yaml.safe_load((ROOT/cfg['evaluation']['acceptance']).read_text());root=ROOT/contract['run']
    baseline=json.loads((root/'baseline_audit/eeg/vq128/summary_v2.json').read_text());continuous=json.loads((root/'baseline_audit/eeg/continuous/summary_v2.json').read_text())
    results={}
    for arm in ('wiener_integer_sum','complex2x2'):
        target=output/arm;target.mkdir(exist_ok=True)
        model=load_model(checkpoint);model.metadata.update(schema='montage_joint_phase_readout_v1',integer_phase_readout=arm,
            phase_config_identity=identity(cfg),phase_source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        if arm=='wiener_integer_sum':
            coefficient=np.empty((3001,2,2),complex)
            for k,b in enumerate(('S','O')):coefficient[:,:,k]=np.asarray(model.transfer[b]).reshape(3001,1)
            model.phase_readout={d:coefficient.copy() for d in model.metadata['positions']}
        else:model.phase_readout=coefficients
        destination=target/'model.npz'
        if not destination.exists():model.save(destination)
        if not (target/'screen_summary.json').exists():
            screen(model,destination,rows,scales,cfg,target,ids_source=ROOT/cfg['source']['screen_ids'],mode_names=('spatial',))
        record=json.loads((target/'screen_summary.json').read_text());candidate=deepcopy(record['summaries']['spatial'])
        candidate.update(modality='eeg',role='screen',checkpoint=str(destination),integer_replay=dict(status='passed' if record['integer_replay_max_abs']==0 else 'failed'))
        gate=score(candidate,baseline,continuous,contract)
        write(target/'reconstruction_acceptance.json',gate)
        export_standard(cfg,destination,target);integer_rate(destination,target)
        results[arm]=dict(macro={k:candidate['metrics'][k]['macro'] for k in ('nmse','rms_gain','correlation','local_rms_correlation','band_4_8_db','band_8_13_db')},
            failed_checks=gate['failed'],replay_max_abs=record['integer_replay_max_abs'])
    write(output/'summary.json',dict(status='completed',experiment_id=cfg['experiment_id'],arms=results,
        elapsed_seconds=time.monotonic()-began,codebook_updates=0,observed_phase_decoder_inputs=0,confirmation_reads=0,test_reads=0,
        pattern_evidence=cfg['source']['pattern_evidence'],unchanged_ID_reason='same frozen encoder and exact saved screen integer arrays',
        verdict='not_accepted' if any(v['failed_checks'] for v in results.values()) else 'numerical_gates_pass_visual_review_pending'))


def phase_review(cfg,output):
    """Replay the original 6000-step hard examples without reselecting them."""
    from experiments.scripts.render_joint_tokenizer_waveforms import render_export
    source=(ROOT/cfg['source']['screen_ids']).parent
    metadata=json.loads((source/'standard_export_v1/waveform_examples.json').read_text())
    packets=json.loads((source/'integer_metadata.json').read_text())
    lookup={(v['row'],v['start_s']):v for v in packets}
    selected=[n for n,item in enumerate(metadata['examples']) if 'candidate_max' in item['selection'] or 'fixed_identity_first' in item['selection']]
    with np.load(source/'standard_export_v1/waveform_examples.npz') as stored:
        target={k:stored[k].copy() for k in ('x','s','o','mask')};original=stored['prediction_S'].copy()
    records={}
    for arm in ('wiener_integer_sum','complex2x2'):
        folder=output/arm/'matched_original_export_v1';folder.mkdir(exist_ok=True)
        checkpoint=output/arm/'model.npz';model=load_model(checkpoint)
        prediction={b:np.zeros_like(target['x'],dtype=float) for b in ('S','O','sum')};replayed={}
        with np.load(ROOT/cfg['source']['screen_ids']) as stored:
            for n,item in enumerate(metadata['examples']):
                packet=lookup[item['row'],item['start_s']];number=packet['crop'];dataset=packet['dataset']
                if number not in replayed:replayed[number]=model.decode_ids({b:stored[f'{b}_{number}'] for b in ('S','O')},dataset)
                native=model.metadata['positions'][dataset];column=next(j for j,name in enumerate(native) if name.upper()==item['position_name'].upper())
                for b in prediction:prediction[b][n]=replayed[number][b][0,column]
        if not (folder/'waveform_examples.npz').exists():
            np.savez_compressed(folder/'waveform_examples.npz',**target,**{'prediction_'+b:v for b,v in prediction.items()})
            copied=deepcopy(metadata)
            for item in copied['examples']:
                item['selection']=['base6000_'+v for v in item['selection']];item['prediction_label']=arm
            copied.update(model=str(checkpoint),prediction_label=arm,selection_source=str(source/'standard_export_v1/waveform_examples.json'))
            write(folder/'waveform_examples.json',copied)
        figures=output/arm/'matched_original_figures_v2'
        if not figures.exists():render_export(folder/'waveform_examples.npz',folder/'waveform_examples.json',figures,indices=selected)
        cases=[]
        for n in selected:
            item=metadata['examples'][n];mask=target['mask'][n].ravel();scale=item['scale'];indices=np.flatnonzero(mask)
            truth=target['s'][n].ravel()*scale;base=original[n].ravel()*scale;estimate=prediction['S'][n].ravel()*scale
            at=int(indices[np.argmax(abs(base[indices]))]);peak=int(indices[np.argmax(abs(estimate[indices]))])
            cases.append(dict(index=n,dataset=item['dataset'],subject=item['subject'],record=item['record'],position=item['position_name'],unit=item['source_unit'],
                RMSE=float(np.sqrt(np.mean((estimate[mask]-truth[mask])**2))),correlation=float(np.corrcoef(estimate[mask],truth[mask])[0,1]),
                original_decoded_peak=dict(time_s=at/200,original_prediction=float(base[at]),target=float(truth[at]),new_prediction=float(estimate[at])),
                new_peak=dict(time_s=peak/200,prediction=float(estimate[peak]),target=float(truth[peak]))))
        records[arm]=cases
    write(output/'matched_original_review.json',dict(status='rendered_for_actual_inspection',selection='all original candidate-worst and first-identity examples; no reselection',
        component='S',examples=records,decoder_inputs='frozen saved integers only; targets read after decoding for this comparison'))


def spatial_phase_probe(cfg,output):
    """Train-subject CV gate for low-rank carriers from other integer electrodes."""
    from src.tokenizers.montage_joint_tokenizer import phase_coefficient,fit_phase_quadratic,lowrank_spatial_residual
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    from src.analysis.dual_component_tokenizer import base_training_aggregate,joint_reconstruction_metrics,joint_reconstruction_summary
    from experiments.scripts.review_joint_tokenizer import score
    if cfg['schema']!='dual_component_montage_cross_phase_v1' or cfg['training']['fitting_splits']!=['train'] or cfg['evaluation']['role']!='screen':
        raise ValueError('train-only terminal spatial-phase contract required')
    if (output/'summary.json').exists():return
    began=time.monotonic();manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT);rows=manifest['rows']
    checkpoint=ROOT/cfg['source']['checkpoint'];base=load_model(checkpoint)
    plan=json.loads((ROOT/cfg['source']['training_plan']).read_text())
    if len(plan)!=62 or any(rows[v['row']]['split']!='train' or rows[v['row']]['modality']!='eeg' for v in plan):raise ValueError('62 train crops before arrays')
    if base.metadata['input_identity']!=manifest['cfg_identity'] or base.metadata['scales_identity']!=manifest['scales_identity']:
        raise ValueError('terminal source G0 coordinates changed before arrays')
    inherited=json.loads((ROOT/cfg['source']['local_statistics']).parent.joinpath('resolved_config.json').read_text())
    if inherited['plan']!=plan or inherited['source_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError('matched local statistics source or physical train plan changed')
    bins=cfg['training']['bin_hz'];edges=list(np.arange(1,45,bins));cache=OrderedDict();items=[]
    output.joinpath('cache').mkdir(exist_ok=True)
    write(output/'resolved_config.json',dict(config=cfg,source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),plan=plan,
        fit='frequency covariance surrogate with invalid targets zeroed; all CV losses use exact original time masks',screen_opened=False))
    for n,crop in enumerate(plan):
        path=output/'cache'/f'{n:03d}.npz';row=rows[crop['row']];dataset=row['dataset'];names=base.metadata['positions'][dataset];p=len(names)
        original=ROOT/cfg['source']['local_statistics']/f'crop_{n:03d}.npz'
        with np.load(original) as s:local={k:s[k].copy() for k in ('S','O','gram','cross','constant','nmse')}
        if path.exists():
            with np.load(path) as s:item={k:s[k].copy() for k in s.files}
        else:
            wave=base.decode_ids({b:local[b] for b in ('S','O')},dataset)
            raw=np.stack([wave[b][0].reshape(p,6000) for b in ('S','O')],axis=-1).transpose(1,0,2).reshape(6000,2*p)
            z=np.fft.rfft(raw,axis=0,norm='ortho')
            examples=[(crop['row'],crop['start_s'],i) for i in crop['positions']];batch=base_tokenizer_batch(rows,examples,scales,array_cache=cache,cache_limit=8)
            native=np.asarray([next(j for j,name in enumerate(names) if name.upper()==row['positions'][i].upper()) for i in crop['positions']])
            target=np.stack([batch[k].reshape(len(examples),6000) for k in ('x','s','o')]);mask=batch['mask'].reshape(len(examples),6000)
            full=np.zeros((6000,p,2));full[:,native,0]=np.where(mask,target[1],0).T;full[:,native,1]=np.where(mask,target[2],0).T
            y=np.fft.rfft(full.reshape(6000,2*p),axis=0,norm='ortho');xx=[];xy=[]
            for lo in edges:
                use=slice(int(lo*30),int(min(lo+bins,45)*30));a=z[use];b=y[use];xx.append(a.conj().T@a/len(a));xy.append(a.conj().T@b/len(a))
            item=dict(z=z,target=target,mask=mask,native=native,examples=np.asarray(examples),
                variance=np.stack([batch['v'+k][:,0,0] for k in ('x','s','o')]),difference=np.stack([batch['dv'+k][:,0,0] for k in ('x','s','o')]),
                input_scales=np.asarray([scales['positions']['eeg'][dataset][rows[crop['row']]['positions'][i]]['scale'] for i in crop['positions']]),xx=np.asarray(xx),xy=np.asarray(xy))
            np.savez_compressed(path,**item)
        item.update(dataset=dataset,subject=row['subject'],record=row['record'],crop=crop,local_gram=local['gram'].mean(0),local_cross=local['cross'].mean(0),local_constant=local['constant'].mean(0),base_nmse=local['nmse'].mean(0));items.append(item)
        write(output/'progress.json',dict(stage='prepare_train_integer_carriers',crops=n+1,planned=62,elapsed_seconds=time.monotonic()-began))
    subjects={}
    for dataset in base.metadata['positions']:
        group=[v for v in items if v['dataset']==dataset]
        fields=dict(xx_real=np.asarray([v['xx'].real for v in group]),xx_imag=np.asarray([v['xx'].imag for v in group]),
            xy_real=np.asarray([v['xy'].real for v in group]),xy_imag=np.asarray([v['xy'].imag for v in group]),
            gram=np.asarray([v['local_gram'] for v in group]),cross=np.asarray([v['local_cross'] for v in group]),constant=np.asarray([v['local_constant'] for v in group]),nmse=np.asarray([v['base_nmse'] for v in group]))
        agg=base_training_aggregate(fields,[(v['crop']['row'],v['crop']['start_s'],v['crop']['positions'][0]) for v in group],rows)['metrics']
        for key in fields:
            for record in agg[key]['by_subject']:subjects.setdefault((dataset,record['subject']),{})[key]=np.asarray(record['value'])
        for key in [k for k in subjects if k[0]==dataset]:
            v=subjects[key];v['xx']=v.pop('xx_real')+1j*v.pop('xx_imag');v['xy']=v.pop('xy_real')+1j*v.pop('xy_imag')
    def local_fit(keys):
        candidates=[]
        for ridge in (.01,.1,1.):
            for smooth in (.1,1.):
                scores=[]
                for fold in range(3):
                    fit=[subjects[k] for j,k in enumerate(keys) if j%3!=fold];held=[subjects[k] for j,k in enumerate(keys) if j%3==fold]
                    if not fit or not held:continue
                    u=fit_phase_quadratic(np.mean([v['gram'] for v in fit],0),np.mean([v['cross'] for v in fit],0),bins,ridge,smooth)
                    scores.append(np.mean([u@v['gram']@u-2*u@v['cross']+v['constant'] for v in held]))
                candidates.append((float(np.mean(scores)),ridge,smooth))
        _,ridge,smooth=min(candidates)
        u=fit_phase_quadratic(np.mean([subjects[k]['gram'] for k in keys],0),np.mean([subjects[k]['cross'] for k in keys],0),bins,ridge,smooth)
        return phase_coefficient(u,bins),dict(ridge=ridge,smooth=smooth)
    def spatial_fit(keys,local,rank,ridge):
        xx=np.mean([subjects[k]['xx'] for k in keys],0);xy=np.mean([subjects[k]['xy'] for k in keys],0);p=xx.shape[-1]//2;models=[];diagnostics=[]
        for b,lo in enumerate(edges):
            near=range(max(0,b-1),min(len(edges),b+2));weight=np.asarray([2. if j==b else 1. for j in near]);weight/=weight.sum()
            covariance=sum(w*xx[j] for w,j in zip(weight,near));cross=sum(w*xy[j] for w,j in zip(weight,near))
            prior=np.kron(np.eye(p),local[int(lo*30)])
            left,right,info=lowrank_spatial_residual(covariance,cross,prior,rank,ridge);models.append((left,right));diagnostics.append(info)
        return models,diagnostics
    def predict(item,local,model):
        z=item['z'];p=z.shape[1]//2;out=np.zeros_like(z)
        for b,lo in enumerate(edges):
            use=slice(int(lo*30),int(min(lo+bins,45)*30));part=z[use];value=np.einsum('fpi,io->fpo',part.reshape(len(part),p,2),local[int(lo*30)]).reshape(len(part),2*p)
            if model is not None:
                left,right=model[b];value+=(part@left)@right
                for position in range(p):
                    use_position=slice(2*position,2*position+2)
                    value[:,use_position]-=part[:,use_position]@(left[use_position]@right[:,use_position])
            out[use]=value
        return np.fft.irfft(out,n=6000,axis=0,norm='ortho').reshape(6000,p,2).transpose(2,1,0)[:,item['native']]
    grid=[(r,l) for r in cfg['training']['ranks'] for l in cfg['training']['ridge_fractions']];records={str(v):[] for v in grid};records['local']=[];fits=[]
    for dataset in base.metadata['positions']:
        keys=sorted([k for k in subjects if k[0]==dataset],key=lambda k:hashlib.sha256(str(k).encode()).hexdigest())
        for fold in range(4):
            training=[k for j,k in enumerate(keys) if j%4!=fold];held={k[1] for j,k in enumerate(keys) if j%4==fold}
            local,local_setting=local_fit(training);models={v:spatial_fit(training,local,*v) for v in grid}
            fits.append(dict(dataset=dataset,fold=fold,held_subjects=sorted(held),local_setting=local_setting,diagnostics={str(v):models[v][1] for v in grid}))
            for item in [v for v in items if v['dataset']==dataset and v['subject'] in held]:
                for candidate in ['local']+grid:
                    out=predict(item,local,None if candidate=='local' else models[candidate][0]);pred=np.stack((out.sum(0),out[0],out[1]));mask=item['mask'];count=mask.sum(1)
                    nmse=np.sum(np.where(mask[None],(pred-item['target'])**2,0),axis=-1)/count[None]/item['variance']
                    gain=np.sqrt(np.sum(np.where(mask[None],pred**2,0),axis=-1)/np.sum(np.where(mask[None],item['target']**2,0),axis=-1))
                    records['local' if candidate=='local' else str(candidate)].append(dict(examples=item['examples'],nmse=nmse.T,rms_gain=gain.T,dataset=dataset,fold=fold))
            write(output/'progress.json',dict(stage='train_subject_CV',dataset=dataset,fold=fold,elapsed_seconds=time.monotonic()-began))
    scores={}
    for candidate,values in records.items():
        examples=np.concatenate([v['examples'] for v in values]);scores[candidate]=base_training_aggregate({k:np.concatenate([v[k] for v in values]) for k in ('nmse','rms_gain')},examples,rows)
    baseline=np.asarray(scores['local']['metrics']['nmse']['macro']);eligible=[v for v in grid if np.all(np.asarray(scores[str(v)]['metrics']['nmse']['macro'])[[0,2]]<=1.05*baseline[[0,2]])]
    best=min(eligible,key=lambda v:(scores[str(v)]['metrics']['nmse']['macro'][1],v[0],-v[1])) if eligible else min(grid,key=lambda v:scores[str(v)]['metrics']['nmse']['macro'][1])
    current=np.asarray(scores[str(best)]['metrics']['nmse']['macro']);wins=[]
    for d in base.metadata['positions']:
        for fold in range(4):
            a=[v['nmse'][:,1].mean() for v in records[str(best)] if v['dataset']==d and v['fold']==fold];b=[v['nmse'][:,1].mean() for v in records['local'] if v['dataset']==d and v['fold']==fold];wins.append(np.mean(a)<np.mean(b))
    dataset_wins=sum(np.asarray(scores[str(best)]['metrics']['nmse']['by_dataset'][d])[1]<np.asarray(scores['local']['metrics']['nmse']['by_dataset'][d])[1] for d in base.metadata['positions'])
    advance=bool(best in eligible and baseline[1]-current[1]>=.02 and current[1]<=.95*baseline[1] and np.mean(wins)>=.75 and dataset_wins>=3)
    audited={key:(defaultdict(list),defaultdict(list),[]) for key in ('local','spatial')}
    for dataset in base.metadata['positions']:
        keys=sorted([k for k in subjects if k[0]==dataset],key=lambda k:hashlib.sha256(str(k).encode()).hexdigest())
        for fold in range(4):
            training=[k for j,k in enumerate(keys) if j%4!=fold];held={k[1] for j,k in enumerate(keys) if j%4==fold};local,_=local_fit(training);model,_=spatial_fit(training,local,*best)
            for item in [v for v in items if v['dataset']==dataset and v['subject'] in held]:
                n=len(item['examples']);batch={'mask':item['mask'].reshape(n,15,400)}
                for k,field in enumerate(('x','s','o')):
                    batch[field]=item['target'][k].reshape(n,15,400);batch['v'+field]=np.broadcast_to(item['variance'][k,:,None,None],(n,15,400));batch['dv'+field]=np.broadcast_to(item['difference'][k,:,None,None],(n,15,400))
                for name,which in [('local',None),('spatial',model)]:
                    prediction=predict(item,local,which);values={'S':prediction[0].reshape(n,15,400),'O':prediction[1].reshape(n,15,400)};values['sum']=values['S']+values['O']
                    result=joint_reconstruction_metrics(values,batch,1,item['input_scales'])
                    for k,v in result['metrics'].items():audited[name][0][k].append(v)
                    for k,v in result['support'].items():audited[name][1][k].append(v)
                    audited[name][2].extend(item['examples'].tolist())
    held_metrics={name:joint_reconstruction_summary({k:np.concatenate(v) for k,v in values[0].items()},{k:np.concatenate(v) for k,v in values[1].items()},values[2],rows) for name,values in audited.items()}
    write(output/'held_train_metrics.json',dict(role='train_subject_holdout',selected_rank=best[0],selected_ridge=best[1],summaries=held_metrics))
    write(output/'training_CV.json',dict(status='completed',scores=scores,selected_rank=best[0],selected_ridge=best[1],fold_fits=fits,
        S_improved_fold_fraction=float(np.mean(wins)),S_improved_datasets=int(dataset_wins),screen_condition_met=advance,
        condition='S improves >=.02 absolute and >=5 percent; >=75 percent outer folds and >=3 datasets; X/O worsen <=5 percent',
        local_baseline='complex2x2 refitted within each outer train fold with inner-train CV; no held labels in local prior'))
    if not advance:
        write(output/'summary.json',dict(status='completed',verdict='train_CV_stopping_rule_triggered',screen_opened=False,
            local_nmse=baseline,spatial_nmse=current,selected_rank=best[0],selected_ridge=best[1],elapsed_seconds=time.monotonic()-began,
            conclusion='no clear repeatable cross-electrode improvement under the declared low-rank strong-ridge family',confirmation_reads=0,test_reads=0));return
    coefficient={}
    for dataset,names in base.metadata['positions'].items():
        keys=[k for k in subjects if k[0]==dataset];local,_=local_fit(keys);model,_=spatial_fit(keys,local,*best)
        blocks=[]
        for b,lo in enumerate(edges):
            residual=model[b][0]@model[b][1]
            for position in range(len(names)):residual[2*position:2*position+2,2*position:2*position+2]=0.
            blocks.append(np.kron(np.eye(len(names)),local[int(lo*30)])+residual)
        coefficient[dataset]=np.asarray(blocks)
    base.phase_readout=coefficient;base.metadata.update(schema='montage_joint_phase_readout_v1',integer_phase_readout='cross_electrode_complex_residual',phase_bin_hz=bins,spatial_phase_config_identity=identity(cfg))
    destination=output/'model.npz';base.save(destination);target=output/'screen';target.mkdir(exist_ok=True)
    screen(base,destination,rows,scales,cfg,target,ids_source=ROOT/cfg['source']['screen_ids'],mode_names=('spatial',));export_standard(cfg,destination,target);integer_rate(destination,target)
    contract=yaml.safe_load((ROOT/cfg['evaluation']['acceptance']).read_text());root=ROOT/contract['run'];record=json.loads((target/'screen_summary.json').read_text());candidate=deepcopy(record['summaries']['spatial'])
    candidate.update(modality='eeg',role='screen',checkpoint=str(destination),integer_replay=dict(status='passed' if record['integer_replay_max_abs']==0 else 'failed'))
    gate=score(candidate,json.loads((root/'baseline_audit/eeg/vq128/summary_v2.json').read_text()),json.loads((root/'baseline_audit/eeg/continuous/summary_v2.json').read_text()),contract);write(target/'reconstruction_acceptance.json',gate)
    write(output/'summary.json',dict(status='completed',verdict='not_accepted' if gate['failed'] else 'numerical_gates_pass_visual_review_pending',screen_opened=True,
        local_CV_nmse=baseline,spatial_CV_nmse=current,screen_macro={k:candidate['metrics'][k]['macro'] for k in ('nmse','rms_gain','correlation','band_4_8_db','band_8_13_db')},failed_checks=gate['failed'],
        elapsed_seconds=time.monotonic()-began,confirmation_reads=0,test_reads=0))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_montage_joint_v1.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','train','evaluate','audit','summarize','phase-readout','phase-review','spatial-phase'],required=True)
    parser.add_argument('--arm',choices=['joint','decoder_control'],default='joint')
    parser.add_argument('--step',type=int,default=1200)
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if args.stage not in ('phase-readout','phase-review','spatial-phase'):validate(cfg)
    torch.set_num_threads(cfg['resources']['numerical_threads'])
    torch.set_num_interop_threads(1)
    args.output.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(cfg['resources']['numerical_threads']):
        if args.stage=='spatial-phase':
            with owner_lock(args.output):spatial_phase_probe(cfg,args.output)
        elif args.stage=='phase-review':
            with owner_lock(args.output,name='review'):phase_review(cfg,args.output)
        elif args.stage=='phase-readout':
            with owner_lock(args.output):phase_readout(cfg,args.output)
        elif args.stage=='prepare':
            with owner_lock(args.output):prepare(cfg,args.output)
        elif args.stage=='train':
            (args.output/args.arm).mkdir(exist_ok=True)
            with owner_lock(args.output/args.arm):train(cfg,args.output,args.arm)
        elif args.stage=='evaluate':
            with owner_lock(args.output/args.arm/f'screen_step_{args.step:04d}',name='evaluation',blocking=True):
                evaluate(cfg,args.output,args.arm,args.step)
        elif args.stage=='audit':
            with owner_lock(args.output/args.arm/f'screen_step_{args.step:04d}',name='audit',blocking=True):
                audit(cfg,args.output,args.arm,args.step)
        else:
            with owner_lock(args.output,name='summary',blocking=True):summarize(cfg,args.output)


if __name__=='__main__':main()
