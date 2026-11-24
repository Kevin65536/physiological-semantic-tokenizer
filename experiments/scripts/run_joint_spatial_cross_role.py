#!/usr/bin/env python3
"""Frozen S70/O25 integer books with paired same-role and offset joint spatial ridge."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from src.analysis.spatial_tokenizer import screen, write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, band_filter, load_model, spectral_features

MODELS = {}


def collect_dataset(task):
    dataset, subjects, rows, scales, cfg, output = task
    destination = output / (dataset + '_fit.json')
    if destination.exists():
        return str(destination)
    names = MODELS['same_role'].metadata['positions'][dataset]
    statistics = {variant: [] for variant in MODELS}
    integers = {variant: {} for variant in MODELS}
    identity = []
    support = []
    cache = OrderedDict()
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        for subject, crops in subjects.items():
            moments = {}
            for variant, model in MODELS.items():
                moments[variant] = []
                for band in range(5):
                    width = sum(int(np.sum(model.roles(dataset, branch) == band)) for branch in ('S', 'O'))
                    moments[variant].append(dict(gram=np.zeros((len(names), width, width)),
                        cross=np.zeros((len(names), width, 2)), count=np.zeros(len(names))))
            for crop in crops:
                fields, observed, valid = base_tokenizer_joint_eeg(rows, crop, scales, names, cache)
                support.append(dict(subject=subject, **crop, observed_positions=int(observed.sum()),
                                    complete_target_positions=int(valid.sum())))
                if not valid.any():
                    continue
                ordinal = len(identity)
                identity.append(dict(crop=ordinal, dataset=dataset, subject=subject, **crop,
                    native_positions=names, complete_target_positions=np.flatnonzero(valid).tolist()))
                targets = [np.stack([band_filter(fields[field], band, axis=0) for field in ('s', 'o')], axis=-1)
                           for band in range(5)]
                x = fields['x'].T.reshape(1, len(names), 15, 400)
                branch_S = None
                for variant, model in MODELS.items():
                    ids = model.encode(x, dataset, observed[None])
                    if branch_S is None:
                        branch_S = ids['S'].copy()
                    else:
                        np.testing.assert_array_equal(branch_S, ids['S'])
                    for branch in ('S', 'O'):
                        integers[variant][f'{branch}_{ordinal}'] = ids[branch]
                    raw = model.decode_ids(ids, dataset, spatial=False)
                    for band in range(5):
                        sources = []
                        for branch in ('S', 'O'):
                            anchors = np.flatnonzero(model.roles(dataset, branch) == band)
                            wave = raw[branch][0].reshape(len(names), 6000).T
                            sources.append(band_filter(wave[:, anchors], band, axis=0))
                        values = np.concatenate(sources, axis=-1)
                        item = moments[variant][band]
                        item['gram'][valid] += values.T @ values / 6000
                        item['cross'][valid] += np.einsum('tm,tpo->pmo', values, targets[band][:, valid], optimize=True) / 6000
                        item['count'] += valid
            for variant in MODELS:
                averaged = []
                for item in moments[variant]:
                    denominator = np.maximum(item['count'], 1)
                    averaged.append(dict(gram=item['gram'] / denominator[:, None, None],
                                         cross=item['cross'] / denominator[:, None, None], valid=item['count'] > 0))
                statistics[variant].append(averaged)
            write(output / (dataset + '_progress.json'), dict(stage='train_joint_spatial_statistics',
                completed_subjects=len(statistics['same_role']), planned_subjects=len(subjects)))
        fits = {}
        for variant, model in MODELS.items():
            payload = {}
            diagnostics = []
            for band in range(5):
                items = [subject[band] for subject in statistics[variant]]
                denominator = np.maximum(np.stack([item['valid'] for item in items]).sum(0), 1)
                gram = np.stack([item['gram'] for item in items]).sum(0) / denominator[:, None, None]
                cross = np.stack([item['cross'] for item in items]).sum(0) / denominator[:, None, None]
                width = gram.shape[-1]
                regularization = cfg['training']['spatial_ridge_fraction'] * np.trace(gram, axis1=1, axis2=2) / width + 1e-12
                system = gram.copy()
                diagonal = np.arange(width)
                system[:, diagonal, diagonal] += regularization[:, None]
                coefficients = np.linalg.solve(system, cross)
                for column, branch in enumerate(('S', 'O')):
                    payload[f'{branch}_{band}'] = coefficients[..., column].T
                diagnostics.append(dict(band=list(BANDS[band]), source_width=width,
                    S_anchor_positions=np.flatnonzero(model.roles(dataset, 'S') == band).tolist(),
                    O_anchor_positions=np.flatnonzero(model.roles(dataset, 'O') == band).tolist(),
                    supported_training_subjects_per_target=denominator.tolist(), ridge=regularization.tolist()))
            np.savez_compressed(output / variant / (dataset + '_weights.npz'), **payload)
            np.savez_compressed(output / variant / (dataset + '_training_ids.npz'), **integers[variant])
            write(output / variant / (dataset + '_training_integer_metadata.json'), identity)
            fits[variant] = diagnostics
    write(output / (dataset + '_support.json'), support)
    write(destination, dict(dataset=dataset, subjects=list(subjects), supported_joint_crops=len(identity), fits=fits))
    return str(destination)


def evaluate(task):
    variant, cfg, rows, scales, output = task
    folder = output / variant
    if (folder / 'screen_summary.json').exists():
        return str(folder / 'screen_summary.json')
    checkpoint = folder / 'model.npz'
    model = load_model(checkpoint)
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        screen(model, checkpoint, rows, scales, cfg, folder)
    return str(folder / 'screen_summary.json')


def synthetic(output):
    rng = np.random.default_rng(77)
    names = list('ABCDE')
    metadata = dict(schema='spatial_waveform_dictionary_v3', codebook_size=4, positions={'fixture': names},
        branch_role_offsets={'S': 0, 'O': 2}, joint_spatial_inputs='both_branch_atoms', output_projection_hz=[1,45])
    centers = {(branch, band): rng.normal(size=(3, spectral_features(np.zeros((1,400)),band).shape[1]))
               for branch in ('S','O') for band in range(5)}
    atoms = {(branch,band): np.r_[np.zeros((1,400)),rng.normal(size=(3,400))]
             for branch in ('S','O') for band in range(5)}
    weights = {('fixture',branch,band): np.zeros((2,5)) for branch in ('S','O') for band in range(5)}
    model = SpatialWaveformDictionary(metadata, {'S':np.ones(3001), 'O':np.ones(3001)}, centers, atoms, weights)
    for band in range(5):
        weights['fixture','S',band][0,np.flatnonzero(model.roles('fixture','S')==band)[0]]=1
        weights['fixture','O',band][1,np.flatnonzero(model.roles('fixture','O')==band)[0]]=1
    ids = {branch:rng.integers(1,4,size=(1,5,15),dtype=np.int32) for branch in ('S','O')}
    independent=model.decode_ids(ids,'fixture',spatial=False)
    prediction=model.decode_ids(ids,'fixture')
    for branch in ('S','O'):
        expected=np.zeros((1,5,6000))
        for position,band in enumerate(model.roles('fixture',branch)):
            expected[:,position]=band_filter(independent[branch][:,position].reshape(1,6000),int(band))
        np.testing.assert_allclose(prediction[branch],expected.reshape(1,5,15,400),atol=1e-12)
    zeros={branch:np.zeros_like(ids[branch]) for branch in ('S','O')}
    for values in model.decode_ids(zeros,'fixture').values():np.testing.assert_array_equal(values,0)
    encoded=model.encode(rng.normal(size=(1,5,15,400)),'fixture',np.ones((1,5),bool))
    assert all(value.shape==(1,5,15) and value.min()>=1 for value in encoded.values())
    model.save(output/'synthetic_model.npz');restored=load_model(output/'synthetic_model.npz')
    for branch,values in prediction.items():np.testing.assert_array_equal(restored.decode_ids(ids,'fixture')[branch],values)
    write(output/'synthetic.json',dict(status='passed',branch_specific_roles=True,explicit_atom_band_linear_map=True,
        missing_ID_zero_contribution=True,checkpoint_integer_replay='exact',integer_count_per_branch_position=15))


def render_matched(cfg, rows, scales, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models={variant:load_model(output/variant/'model.npz') for variant in cfg['model']['variants']}
    metadata=json.loads((output/'same_role/integer_metadata.json').read_text())
    if metadata!=json.loads((output/'offset_two/integer_metadata.json').read_text()):
        raise ValueError('matched spatial role contexts required')
    with np.load(output/'offset_two/spatial_metrics.npz') as stored:
        examples=stored['examples'];error=stored['nmse'].mean(axis=(1,2))
    choices={}
    for item in metadata:
        dataset=item['dataset']
        choices.setdefault(dataset,dict(fixed=item))
    for dataset,choice in choices.items():
        indices=[i for i,(row_index,_,_) in enumerate(examples) if rows[row_index]['dataset']==dataset]
        worst=max(indices,key=lambda i:error[i] if np.isfinite(error[i]) else float('inf'))
        row_index,start,_=examples[worst]
        choice['worst']=next(item for item in metadata if item['row']==row_index and item['start_s']==start)
        choice['worst_panel_index']=worst
    integers={variant:np.load(output/variant/'integer_ids.npz') for variant in models}
    cache=OrderedDict();identities=[]
    for dataset,choice in choices.items():
        for selection in ('fixed','worst'):
            item=choice[selection];row_index,start=item['row'],item['start_s'];row=rows[row_index]
            if row['split']!='validation':raise ValueError('validation visualization scope required')
            panel_index=choice['worst_panel_index'] if selection=='worst' else int(np.flatnonzero((examples[:,0]==row_index)&(examples[:,1]==start))[0])
            native_position=int(examples[panel_index,2]);name=row['positions'][native_position]
            position=next(i for i,value in enumerate(item['native_positions']) if value.upper()==name.upper())
            batch=base_tokenizer_batch(rows,[(row_index,start,native_position)],scales,array_cache=cache,cache_limit=8)
            predictions={}
            for variant,model in models.items():
                codes={branch:integers[variant][f"{branch}_{item['crop']}"] for branch in ('S','O')}
                predictions[variant]=model.decode_ids(codes,dataset)
            scale=scales['positions']['eeg'][dataset][name]['scale'];t=np.arange(6000)/200
            fig,axes=plt.subplots(3,3,figsize=(16,8),constrained_layout=True)
            for index,(branch,field) in enumerate((('S','s'),('O','o'),('sum','x'))):
                truth=batch[field][0].reshape(-1)*scale
                curves={variant:values[branch][0,position].reshape(-1)*scale for variant,values in predictions.items()}
                for column,bounds in enumerate(((0,30),(14,16),(13.5,14.5))):
                    ax=axes[index,column];ax.plot(t,truth,color='#202020',linewidth=.8,label='target')
                    for variant,color,style in [('same_role','#3977ad','--'),('offset_two','#d28220','-')]:
                        ax.plot(t,curves[variant],color=color,linestyle=style,linewidth=.8,label=variant)
                    visible=(t>=bounds[0])&(t<bounds[1]);all_values=[truth]+list(curves.values())
                    low=min(float(values[visible].min()) for values in all_values);high=max(float(values[visible].max()) for values in all_values)
                    margin=.05*max(high-low,1e-20);ax.set_ylim(low-margin,high+margin);ax.set_xlim(*bounds)
                    for boundary in range(2,30,2):ax.axvline(boundary,color='grey',linestyle=':',linewidth=.5)
                    invalid=~batch['mask'][0].reshape(-1);transitions=np.diff(np.r_[False,invalid,False].astype(int))
                    for begin,end in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
                        ax.axvspan(begin/200,end/200,color='grey',alpha=.25)
                    ax.set_xlabel('Window time (s)');ax.set_ylabel(('X=S+O' if branch=='sum' else branch)+' ('+row['source_unit']+')')
                    if index==0 and column==0:ax.legend(fontsize=8)
            fig.suptitle(f"Frozen S70/O25 K512 / {selection} / {dataset} / {row['subject']} / {row['record']} / {name} / crop {start}s\nFull native-layout integers; dotted 2s boundaries; grey unscored samples")
            path=output/(dataset+'_'+selection+'_matched_waveform.png');fig.savefig(path,dpi=200);plt.close(fig)
            identities.append(dict(selection=selection,dataset=dataset,subject=row['subject'],record=row['record'],parent=row['parent'],
                manifest_row=row_index,native_start_sample=row['native_start_sample'],native_rate_hz=row['native_rate_hz'],
                crop_start_s=start,position=name,panel_index=panel_index,figure=str(path),
                rule='first registered screen crop and selected position' if selection=='fixed' else 'largest offset-two per-example mean XSO NMSE within dataset'))
    for stored in integers.values():stored.close()
    write(output/'matched_visual_examples.json',identities)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_cross_role_v1.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.synthetic:synthetic(args.output);return
    cfg=yaml.safe_load(args.config.read_text())
    if (cfg['schema']!='dual_component_spatial_cross_role_v1' or cfg['model']['variants']!={'same_role':{'S':0,'O':0},'offset_two':{'S':0,'O':2}}
            or cfg['training']['fitting_splits']!=['train'] or cfg['evaluation']['role']!='screen'
            or cfg['model']['unencoded_decoder_inputs'] or cfg['model']['codebook_size']!=512):
        raise ValueError('fixed train-only S70/O25 paired branch-role contract required')
    if (args.output/'summary.json').exists() and not args.render_only:raise FileExistsError('completed structural comparison retained')
    started=time.monotonic();manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT);rows=manifest['rows']
    if args.render_only:
        render_matched(cfg,rows,scales,args.output)
        return
    sources={branch:load_model(ROOT/cfg['source'][branch]) for branch in ('S','O')}
    reference=sources['S'];metadata=json.loads(json.dumps(reference.metadata))
    for branch,source in sources.items():
        if (source.metadata['input_identity']!=manifest['cfg_identity'] or source.metadata['scales_identity']!=manifest['scales_identity']
                or source.metadata['positions']!=metadata['positions'] or source.codebook_size!=512
                or source.metadata['semantic_distance_fraction']!=cfg['model']['branch_semantic_distance_fraction'][branch]):
            raise ValueError('frozen per-branch source identity changed before arrays')
    plan=json.loads((ROOT/cfg['source']['training_plan']).read_text())['plan']
    if any(rows[crop['row']]['split']!='train' for subjects in plan.values() for crops in subjects.values() for crop in crops):
        raise ValueError('training-only spatial plan before signal reads')
    source_sha={branch:hashlib.sha256((ROOT/cfg['source'][branch]).read_bytes()).hexdigest() for branch in sources}
    resolved=dict(config=cfg,source_sha256=source_sha,input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'],training_plan=plan)
    path=args.output/'resolved_config.json'
    if path.exists() and json.loads(path.read_text())!=resolved:raise ValueError('retained comparison configuration changed')
    write(path,resolved)
    for variant,offsets in cfg['model']['variants'].items():
        folder=args.output/variant;folder.mkdir(exist_ok=True)
        current=dict(metadata,schema='spatial_waveform_dictionary_v3',branch_role_offsets=offsets,joint_spatial_inputs='both_branch_atoms',
            branch_dictionary_sources=cfg['source'],branch_dictionary_sha256=source_sha,
            semantic_distance_fraction=cfg['model']['branch_semantic_distance_fraction'],
            decoder_branch_dependencies={'S':['S_IDs','O_IDs'],'O':['S_IDs','O_IDs']},
            spatial_ridge_fraction=cfg['training']['spatial_ridge_fraction'])
        fields={name:{(branch,band):getattr(sources[branch],name)[branch,band] for branch in ('S','O') for band in range(5)}
                for name in ('centers','atoms','feature_weights')}
        temporary=SpatialWaveformDictionary(current,{branch:sources[branch].transfer[branch] for branch in ('S','O')},weights={},**fields)
        temporary.weights={(dataset,branch,band):np.zeros((sum(int(np.sum(temporary.roles(dataset,b)==band)) for b in ('S','O')),len(names)))
            for dataset,names in metadata['positions'].items() for branch in ('S','O') for band in range(5)}
        MODELS[variant]=temporary
    tasks=[(dataset,subjects,rows,scales,cfg,args.output) for dataset,subjects in plan.items()]
    with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        list(pool.map(collect_dataset,tasks))
    parameter_counts={}
    for variant,model in MODELS.items():
        for dataset in plan:
            with np.load(args.output/variant/(dataset+'_weights.npz')) as archive:
                for branch in ('S','O'):
                    for band in range(5):model.weights[dataset,branch,band]=archive[f'{branch}_{band}']
        model.save(args.output/variant/'model.npz')
        parameter_counts[variant]=sum(value.size for value in model.weights.values())
    if len(set(parameter_counts.values()))!=1:raise ValueError('joint spatial parameter capacity must match')
    tasks=[(variant,cfg,rows,scales,args.output) for variant in MODELS]
    with ProcessPoolExecutor(max_workers=cfg['resources']['screen_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
        paths=list(pool.map(evaluate,tasks))
    scores={}
    for path in paths:
        path=Path(path);metrics=json.loads(path.read_text())['summaries']['spatial']['metrics']
        scores[path.parent.name]={name:np.asarray(metrics[name]['macro']).ravel() for name in
            ['nmse','rms_gain','correlation','band_4_8_abs_db','band_8_13_abs_db','local_rms_correlation']}
    gain=float(scores['same_role']['nmse'][1]-scores['offset_two']['nmse'][1])
    bands_preserved=all(scores['offset_two'][key][1]<=scores['same_role'][key][1] for key in ['band_4_8_abs_db','band_8_13_abs_db'])
    write(args.output/'summary.json',dict(status='completed',component_order=['X','S','O'],scores=scores,
        joint_spatial_parameters=parameter_counts,offset_absolute_S_NMSE_gain=gain,
        substantial_structural_gain=bool(gain>=.02 and bands_preserved),elapsed_seconds=time.monotonic()-started,
        acceptance='original joint reconstruction, pattern, visual, replay and confirmation gates remain required'))


if __name__=='__main__':main()
